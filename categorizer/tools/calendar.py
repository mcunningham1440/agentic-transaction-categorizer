"""search_calendar tool: read events from the user's primary Google Calendar."""

import asyncio
import threading
from datetime import datetime, time, timezone

from categorizer.archive.google_auth import build_calendar_service
from categorizer.tools._common import parse_iso_date, run_blocking

_calendar_service = None
_calendar_service_lock = asyncio.Lock()

# Serializes the threaded .execute() call below. The cached service's underlying
# httplib2.Http is not thread-safe, but search_calendar_sync runs in a thread
# pool (run_blocking) under concurrent eval load; without this, two threads
# sharing the one TLS connection corrupt the stream ("SSLError: record layer
# failure"). Mirrors _gmail_api_lock in tools/gmail.py. The asyncio
# _calendar_service_lock above only guards lazy init, not the .execute() call.
_calendar_api_lock = threading.Lock()


async def get_calendar_service():
    global _calendar_service
    async with _calendar_service_lock:
        if _calendar_service is None:
            _calendar_service = await run_blocking(build_calendar_service)
    return _calendar_service


def search_calendar_sync(service, start_date: str, end_date: str) -> str:
    start_dt = parse_iso_date(start_date)
    end_dt = parse_iso_date(end_date)

    time_min = datetime.combine(start_dt.date(), time.min, tzinfo=timezone.utc).isoformat()
    time_max = datetime.combine(end_dt.date(), time.max, tzinfo=timezone.utc).isoformat()

    with _calendar_api_lock:
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
