"""Agent tools (LangChain) and the list passed to create_agent.

Each tool is a thin `@tool`-decorated async wrapper over a sync implementation
in its sibling module (messages / calendar / web); the sync work runs in a
thread via `run_blocking` so it doesn't block the event loop. The terminal
"commit a category" step is no longer a tool — it is handled by the agent's
structured output (`response_format`) in `categorizer.agent`.

Each wrapper catches its own exceptions and returns a JSON `{"error": ...}`
string rather than raising. This is deliberate (preserved from the previous
`dispatch_tool`): a single tool failure degrades to an error payload the model
can read and react to, instead of aborting the whole categorization.
"""

import json

from langchain.tools import tool

from categorizer.tools._common import run_blocking
from categorizer.tools.calendar import get_calendar_service, search_calendar_sync
from categorizer.tools.messages import search_messages_sync
from categorizer.tools.web import web_search_sync


@tool
async def search_messages(keyword: str, start_date: str, end_date: str) -> str:
    """Keyword search the user's iMessage database for messages whose text contains the keyword, within an ISO date range.

    Args:
        keyword: Substring to match within message text (case-insensitive).
        start_date: Inclusive start date, ISO format YYYY-MM-DD.
        end_date: Inclusive end date, ISO format YYYY-MM-DD.
    """
    try:
        return await run_blocking(search_messages_sync, keyword, start_date, end_date)
    except Exception as e:  # noqa: BLE001 - degrade to error payload, see module docstring
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


@tool
async def search_calendar(start_date: str, end_date: str) -> str:
    """Return all events on the user's primary Google Calendar within an ISO date range.

    Args:
        start_date: Inclusive start date, ISO format YYYY-MM-DD.
        end_date: Inclusive end date, ISO format YYYY-MM-DD.
    """
    try:
        service = await get_calendar_service()
        return await run_blocking(search_calendar_sync, service, start_date, end_date)
    except Exception as e:  # noqa: BLE001 - degrade to error payload, see module docstring
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


@tool
async def web_search(query: str) -> str:
    """Search the web via Perplexity for information about a merchant, brand, or business. Useful for identifying unfamiliar transaction names.

    Args:
        query: The search query.
    """
    try:
        return await run_blocking(web_search_sync, query)
    except Exception as e:  # noqa: BLE001 - degrade to error payload, see module docstring
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


# Non-terminal tools the agent may consult. The terminal commit is structured
# output, not a tool, so it is not in this list.
AGENT_TOOLS = [search_messages, search_calendar, web_search]
TOOL_NAMES = [t.name for t in AGENT_TOOLS]
