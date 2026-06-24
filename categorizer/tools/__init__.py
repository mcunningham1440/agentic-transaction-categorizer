"""Agent tool definitions and the dispatcher that runs them.

The OpenAI tool schemas (TOOL_DEFINITIONS) and the dispatch_tool router live
here; each tool's implementation lives in its own sibling module
(messages / calendar / web). The terminal categorize_transaction tool is handled
by the agent loop directly, not dispatched here.
"""

import json

from categorizer.categories import CATEGORIES
from categorizer.tools._common import run_blocking
from categorizer.tools.calendar import get_calendar_service, search_calendar_sync
from categorizer.tools.messages import search_messages_sync
from categorizer.tools.web import web_search_sync


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


async def dispatch_tool(name: str, args: dict) -> str:
    """Run a non-terminal tool and return a string payload for function_call_output.

    Terminal `categorize_transaction` is handled by the agent loop directly,
    not here.
    """
    try:
        if name == "search_messages":
            return await run_blocking(
                search_messages_sync,
                args["keyword"],
                args["start_date"],
                args["end_date"],
            )
        if name == "search_calendar":
            service = await get_calendar_service()
            return await run_blocking(
                search_calendar_sync,
                service,
                args["start_date"],
                args["end_date"],
            )
        if name == "web_search":
            return await run_blocking(web_search_sync, args["query"])
        return json.dumps({"error": f"unknown tool: {name}"})
    except KeyError as e:
        return json.dumps({"error": f"missing required argument: {e.args[0]}"})
    except ValueError as e:
        return json.dumps({"error": f"bad argument: {e}"})
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
