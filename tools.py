import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, time, timezone

from perplexity import Perplexity


CATEGORIES = (
    "Other bills",
    "Dining alone",
    "Dining social",
    "Cafeteria",
    "Gas",
    "Groceries",
    "Social recreation",
    "Solo recreation",
    "Drinks",
    "Parking",
    "Tolls",
    "Gifts",
    "Clothes, acessories, haircuts, etc.",
    "Healthcare",
    "Education and career",
    "Public transit + rideshare",
    "Car maintenance",
    "Miscellaneous",
    "Rent",
    "Vacation",
    "Budgeted exceptional expenses",
    "Loan repayment",
    "Internal transfer",
    "Income",
    "Extra income",
)

IMESSAGE_DB_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "imessage_search", "chat.db")
)
GOOGLE_CREDENTIALS_PATH = os.path.join(os.path.dirname(__file__), "credentials.json")
GOOGLE_TOKEN_PATH = os.path.join(os.path.dirname(__file__), "token.json")
GOOGLE_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")

TRUNCATION_LIMIT = 6000


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "categorize_transaction",
        "description": (
            "Commit a final category for the transaction"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": list(CATEGORIES),
                    "description": "The chosen category. Must be one of the listed values",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief justification for the chosen category",
                },
            },
            "required": ["category", "reasoning"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_messages",
        "description": (
            "Keyword search the user's iMessage database for messages whose text "
            "contains the keyword, within an ISO date range. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "Substring to match within message text (case-insensitive).",
                },
                "start_date": {
                    "type": "string",
                    "description": "Inclusive start date, ISO format YYYY-MM-DD.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Inclusive end date, ISO format YYYY-MM-DD.",
                },
            },
            "required": ["keyword", "start_date", "end_date"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_calendar",
        "description": (
            "Return all events on the user's primary Google Calendar within an ISO date range."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Inclusive start date, ISO format YYYY-MM-DD.",
                },
                "end_date": {
                    "type": "string",
                    "description": "Inclusive end date, ISO format YYYY-MM-DD.",
                },
            },
            "required": ["start_date", "end_date"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "web_search",
        "description": (
            "Search the web via Perplexity for information about a merchant, "
            "brand, or business. Useful for identifying unfamiliar transaction "
            "names."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _parse_iso_date(s: str) -> datetime:
    # Strict ISO YYYY-MM-DD. Anything else raises ValueError and the caller
    # surfaces the error back to the model as a tool result.
    return datetime.strptime(s, "%Y-%m-%d")


def _truncate(payload: str) -> str:
    if len(payload) <= TRUNCATION_LIMIT:
        return payload
    return payload[:TRUNCATION_LIMIT] + "...[truncated]"


_ATTRIBUTED_BODY_RE = re.compile(rb"\x94\x84.\x2b(.)")


def _extract_attributed_body_text(blob: bytes) -> str:
    # Mirrors imessage_search/export_char_ratio.py:6-11 — locates the inline
    # UTF-8 text inside the NSTypedStream blob used when m.text is NULL.
    m = _ATTRIBUTED_BODY_RE.search(blob)
    if not m:
        return ""
    length = m.group(1)[0]
    start = m.end()
    return blob[start : start + length].decode("utf-8", errors="replace")


def _search_messages_sync(keyword: str, start_date: str, end_date: str) -> str:
    _parse_iso_date(start_date)
    _parse_iso_date(end_date)

    if not os.path.exists(IMESSAGE_DB_PATH):
        return json.dumps({"error": f"iMessage database not found at {IMESSAGE_DB_PATH}"})

    like_pattern = f"%{keyword.lower()}%"
    conn = sqlite3.connect(f"file:{IMESSAGE_DB_PATH}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                datetime(m.date/1000000000 + strftime('%s', '2001-01-01'), 'unixepoch') AS ts,
                m.is_from_me,
                m.text,
                m.attributedBody,
                c.chat_identifier
            FROM message m
            JOIN chat_message_join cmj ON m.rowid = cmj.message_id
            JOIN chat c ON cmj.chat_id = c.rowid
            WHERE date(ts) BETWEEN ? AND ?
              AND (
                LOWER(m.text) LIKE ?
                OR (m.text IS NULL AND m.attributedBody IS NOT NULL)
              )
            ORDER BY m.date
            """,
            (start_date, end_date, like_pattern),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    kw_lower = keyword.lower()
    hits = []
    for ts, is_from_me, text, blob, contact in rows:
        if text:
            body = text
        elif blob:
            body = _extract_attributed_body_text(blob)
            if kw_lower not in body.lower():
                continue
        else:
            continue
        hits.append({
            "timestamp": ts,
            "direction": "sent" if is_from_me else "received",
            "contact": contact,
            "text": body,
        })

    return _truncate(json.dumps({"count": len(hits), "messages": hits}, ensure_ascii=False))


_calendar_service = None
_calendar_service_lock = asyncio.Lock()


def _build_calendar_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if os.path.exists(GOOGLE_TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_PATH, GOOGLE_SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(GOOGLE_CREDENTIALS_PATH):
                raise FileNotFoundError(
                    f"Google OAuth client secrets not found at {GOOGLE_CREDENTIALS_PATH}. "
                    "Create an OAuth client at console.cloud.google.com (Calendar API enabled), "
                    "download the JSON, and save it to that path."
                )
            flow = InstalledAppFlow.from_client_secrets_file(GOOGLE_CREDENTIALS_PATH, GOOGLE_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(GOOGLE_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds, cache_discovery=False)


async def _run_blocking(fn, *args):
    # asyncio.to_thread was added in Python 3.9; this project runs on 3.8.
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args))


async def _get_calendar_service():
    global _calendar_service
    async with _calendar_service_lock:
        if _calendar_service is None:
            _calendar_service = await _run_blocking(_build_calendar_service)
    return _calendar_service


def _search_calendar_sync(service, start_date: str, end_date: str) -> str:
    start_dt = _parse_iso_date(start_date)
    end_dt = _parse_iso_date(end_date)

    time_min = datetime.combine(start_dt.date(), time.min, tzinfo=timezone.utc).isoformat()
    time_max = datetime.combine(end_dt.date(), time.max, tzinfo=timezone.utc).isoformat()

    events_result = service.events().list(
        calendarId="primary",
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True,
        orderBy="startTime",
        maxResults=100,
    ).execute()

    events = []
    for ev in events_result.get("items", []):
        events.append({
            "start": ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date"),
            "end": ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date"),
            "summary": ev.get("summary"),
            "location": ev.get("location"),
            "description": ev.get("description"),
        })

    return _truncate(json.dumps({"count": len(events), "events": events}, ensure_ascii=False))


def _web_search_sync(query: str) -> str:
    if not PERPLEXITY_API_KEY:
        return json.dumps({"error": "PERPLEXITY_API_KEY env var not set"})

    client = Perplexity(api_key=PERPLEXITY_API_KEY)
    search = client.search.create(query=[query])

    if not search.results:
        return json.dumps({"count": 0, "results": []})

    today = datetime.now().date()
    results = []
    for r in search.results:
        item = {
            "title": r.title or "",
            "url": r.url or "",
            "snippet": getattr(r, "snippet", None) or "",
        }
        r_date = getattr(r, "date", None)
        if r_date:
            try:
                d = datetime.strptime(r_date, "%Y-%m-%d").date()
                item["date"] = r_date
                item["days_ago"] = (today - d).days
            except ValueError:
                item["date"] = r_date
        results.append(item)

    return _truncate(json.dumps({"count": len(results), "results": results}, ensure_ascii=False))


async def dispatch_tool(name: str, args: dict) -> str:
    """Run a non-terminal tool and return a string payload for function_call_output.

    Terminal `categorize_transaction` is handled by the agent loop directly,
    not here.
    """
    try:
        if name == "search_messages":
            return await _run_blocking(
                _search_messages_sync,
                args["keyword"],
                args["start_date"],
                args["end_date"],
            )
        if name == "search_calendar":
            service = await _get_calendar_service()
            return await _run_blocking(
                _search_calendar_sync,
                service,
                args["start_date"],
                args["end_date"],
            )
        if name == "web_search":
            return await _run_blocking(_web_search_sync, args["query"])
        return json.dumps({"error": f"unknown tool: {name}"})
    except KeyError as e:
        return json.dumps({"error": f"missing required argument: {e.args[0]}"})
    except ValueError as e:
        return json.dumps({"error": f"bad argument: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
