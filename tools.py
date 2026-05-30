import asyncio
import json
import os
import sqlite3
from datetime import datetime, time, timezone

from perplexity import Perplexity

from google_auth import build_calendar_service


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

IMESSAGE_DB_PATH = os.path.expanduser("~/Library/Messages/chat.db")

PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


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


_NSSTRING_MARKER = b"NSString"


def _extract_attributed_body_text(blob: bytes) -> str:
    # The attributedBody column holds an NSMutableAttributedString serialized in
    # Apple's typedstream format. The message text is a length-prefixed UTF-8 run
    # that follows the NSString class marker and a 0x2b ("+") type tag. We anchor
    # on the NSString marker rather than the surrounding object bytes: those
    # bytes include a typedstream back-reference index (0x94, 0x95, ...) that
    # increments as the stream reuses the class, so anchoring on a fixed value
    # like \x94 silently dropped every message that happened to land on a
    # different index (~13-32% of texts in practice).
    #
    # After the 0x2b tag the run length is a typedstream variable-length integer:
    # a single byte < 0x81 is the length itself, while a 0x81 tag means the next
    # 2 bytes (little-endian) hold it. (0x82/0x83 forms exist for larger values
    # but a single message never reaches them.)
    marker = blob.find(_NSSTRING_MARKER)
    if marker == -1:
        return ""
    # First 0x2b after the marker; +8 skips the marker so a stray match inside
    # the literal "NSString" can't be picked up (there isn't one, but be exact).
    plus = blob.find(b"\x2b", marker + len(_NSSTRING_MARKER))
    if plus == -1:
        return ""
    i = plus + 1
    if i >= len(blob):
        return ""
    tag = blob[i]
    if tag == 0x81:
        if i + 3 > len(blob):
            return ""
        length = int.from_bytes(blob[i + 1 : i + 3], "little")
        start = i + 3
    else:
        length = tag
        start = i + 1
    return blob[start : start + length].decode("utf-8", errors="replace")


MESSAGES_MAX_LEN = 20000


def _search_messages_sync(keyword: str, start_date: str, end_date: str) -> str:
    _parse_iso_date(start_date)
    _parse_iso_date(end_date)

    if not os.path.exists(IMESSAGE_DB_PATH):
        raise FileNotFoundError(f"iMessage database not found at {IMESSAGE_DB_PATH}")

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
                m.attributedBody
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
    blocks = []
    wrapper_overhead = len('<messages incomplete="true">\n\n</messages>')
    content_len = 0
    incomplete = False
    for ts, is_from_me, text, blob in rows:
        if text:
            body = text
        elif blob:
            body = _extract_attributed_body_text(blob)
            if kw_lower not in body.lower():
                continue
        else:
            continue

        direction = "sent" if is_from_me else "received"
        block = (
            "  <message>\n"
            f"    <datetime>{ts}</datetime>\n"
            f"    <direction>{direction}</direction>\n"
            f"    <text>{body}</text>\n"
            "  </message>"
        )
        addition = len(block) + (1 if blocks else 0)
        if wrapper_overhead + content_len + addition > MESSAGES_MAX_LEN:
            incomplete = True
            break
        blocks.append(block)
        content_len += addition

    attr = ' incomplete="true"' if incomplete else ""
    advisory = (
        "\n  <advisory>Results were truncated before all matching messages "
        "could be returned. Narrow the date range or use a more specific "
        "keyword and search again.</advisory>"
        if incomplete
        else ""
    )
    if not blocks:
        return f"<messages{attr}>{advisory}</messages>" if advisory else f"<messages{attr}></messages>"
    inner = "\n".join(blocks)
    return f"<messages{attr}>\n{inner}{advisory}\n</messages>"


_calendar_service = None
_calendar_service_lock = asyncio.Lock()


async def _run_blocking(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args))


async def _get_calendar_service():
    global _calendar_service
    async with _calendar_service_lock:
        if _calendar_service is None:
            _calendar_service = await _run_blocking(build_calendar_service)
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

    items = events_result.get("items", [])
    if not items:
        return "<events></events>"

    blocks = []
    for ev in items:
        start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date") or ""
        end = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date") or ""
        summary = ev.get("summary") or ""
        location = ev.get("location") or ""
        description = ev.get("description") or ""
        blocks.append(
            "  <event>\n"
            f"    <start>{start}</start>\n"
            f"    <end>{end}</end>\n"
            f"    <summary>{summary}</summary>\n"
            f"    <location>{location}</location>\n"
            f"    <description>{description}</description>\n"
            "  </event>"
        )

    inner = "\n".join(blocks)
    return f"<events>\n{inner}\n</events>"


def _web_search_sync(query: str) -> str:
    if not PERPLEXITY_API_KEY:
        return "<error>PERPLEXITY_API_KEY env var not set</error>"

    client = Perplexity(api_key=PERPLEXITY_API_KEY)
    search = client.search.create(
        query=[query],
        max_results=5,
        max_tokens_per_page=1024,
        )

    if not search.results:
        return "<results></results>"

    blocks = []
    for i, r in enumerate(search.results, start=1):
        title = r.title or ""
        url = r.url or ""
        snippet = getattr(r, "snippet", None) or ""
        blocks.append(
            f'  <result index="{i}">\n'
            f"    <title>{title}</title>\n"
            f"    <url>{url}</url>\n"
            f"    <snippet>{snippet}</snippet>\n"
            f"  </result>"
        )

    inner = "\n".join(blocks)
    return f"<results>\n{inner}\n</results>"


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
