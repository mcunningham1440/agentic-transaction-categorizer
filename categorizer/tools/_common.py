"""Helpers shared across tool implementations."""

import asyncio
from datetime import datetime


def parse_iso_date(s: str) -> datetime:
    # Strict ISO YYYY-MM-DD. Anything else raises ValueError and the caller
    # surfaces the error back to the model as a tool result.
    return datetime.strptime(s, "%Y-%m-%d")


async def run_blocking(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args))
