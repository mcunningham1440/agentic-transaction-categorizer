"""search_gmail tool: keyword-search the user's Gmail within a date range,
returning headers plus a length-bounded plain-text excerpt of each body."""

import asyncio
import base64
import re
import threading
from datetime import timedelta

from categorizer.archive.google_auth import build_gmail_service
from categorizer.tools._common import parse_iso_date, run_blocking

_gmail_service = None
_gmail_service_lock = asyncio.Lock()

# The cached service's underlying httplib2.Http is NOT thread-safe, but
# search_gmail_sync runs in a thread pool (run_blocking) and the eval issues
# transactions concurrently. Two threads hitting the one shared connection at
# once corrupts the TLS stream ("SSLError: record layer failure"). This lock
# serializes the actual HTTP requests so only one is in flight at a time. Gmail
# call volume is low, so the serialization cost is negligible. (The asyncio
# _gmail_service_lock above only guards lazy init in the async layer, not these
# threaded .execute() calls.)
_gmail_api_lock = threading.Lock()

# Per-email body excerpt cap. Bodies are multipart HTML with a lot of footer
# boilerplate; without a cap a handful of emails would balloon the token cost.
# FLAG: this truncates text fed to the LLM — if an email's useful content (a
# product name, an order total) sits past this many characters it is silently
# lost. For Amazon order emails the order line is near the top, so a top-of-body
# cap is usually safe, but "usually" is the risk.
BODY_CHAR_CAP = 1000


async def get_gmail_service():
    global _gmail_service
    async with _gmail_service_lock:
        if _gmail_service is None:
            _gmail_service = await run_blocking(build_gmail_service)
    return _gmail_service


def _decode(data: str) -> str:
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def _collect(part, plain, html):
    mime = part.get("mimeType", "")
    data = part.get("body", {}).get("data")
    if data:
        if mime == "text/plain":
            plain.append(_decode(data))
        elif mime == "text/html":
            html.append(_decode(data))
    for sub in part.get("parts", []) or []:
        _collect(sub, plain, html)


def _extract_body(payload) -> str:
    # Prefer text/plain (already human-readable); fall back to stripping tags
    # from text/html. Whitespace is collapsed so the cap measures real content,
    # not the markup indentation Amazon emails are padded with.
    plain, html = [], []
    _collect(payload, plain, html)
    if plain:
        text = "\n".join(plain)
    elif html:
        text = re.sub(r"<[^>]+>", " ", "\n".join(html))
    else:
        text = ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > BODY_CHAR_CAP:
        text = text[:BODY_CHAR_CAP] + "…"
    return text


def search_gmail_sync(service, keyword: str, start_date: str, end_date: str) -> str:
    start_dt = parse_iso_date(start_date)
    # Gmail's before: operator is exclusive, so add a day to keep end_date
    # inclusive (mirrors search_calendar's use of time.max for the end day).
    end_dt = parse_iso_date(end_date) + timedelta(days=1)

    # keyword is left unquoted so multiple space-separated terms AND together
    # (Gmail's default), letting the agent combine e.g. a merchant with a dollar
    # amount. The trade-off is that a multi-word merchant name is matched as
    # separate AND-ed terms rather than an exact phrase.
    q = (
        f"{keyword} "
        f"after:{start_dt.strftime('%Y/%m/%d')} "
        f"before:{end_dt.strftime('%Y/%m/%d')}"
    )

    # messages.list returns only {id, threadId} per hit, so each id needs a
    # separate messages.get to retrieve the body. Each .execute() is serialized
    # by _gmail_api_lock (see note at module level) to keep concurrent threads
    # off the shared, non-thread-safe HTTP connection.
    with _gmail_api_lock:
        resp = service.users().messages().list(
            userId="me",
            q=q,
            maxResults=20,
        ).execute()

    ids = [m["id"] for m in resp.get("messages", [])]
    if not ids:
        return "<emails></emails>"

    blocks = []
    for mid in ids:
        with _gmail_api_lock:
            msg = service.users().messages().get(
                userId="me",
                id=mid,
                format="full",
            ).execute()
        headers = {h["name"]: h["value"]
                   for h in msg.get("payload", {}).get("headers", [])}
        body = _extract_body(msg.get("payload", {}))
        blocks.append(
            "  <email>\n"
            f"    <date>{headers.get('Date', '')}</date>\n"
            f"    <from>{headers.get('From', '')}</from>\n"
            f"    <subject>{headers.get('Subject', '')}</subject>\n"
            f"    <body>{body}</body>\n"
            "  </email>"
        )

    inner = "\n".join(blocks)
    return f"<emails>\n{inner}\n</emails>"
