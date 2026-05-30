import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import pandas as pd
from openai import AsyncOpenAI

from archive_index import top_n_similar
from tools import CATEGORIES, TOOL_DEFINITIONS, dispatch_tool


MODEL = "gpt-5.4-mini"
REASONING_EFFORT = "low"
MAX_ROUNDS = 10

MODEL_PRICING = {
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
}


def compute_cost(model: str, input_tokens: int, cached_input_tokens: int,
                 output_tokens: int) -> dict:
    """USD cost breakdown for one model's token usage.

    Returns {'input_cost', 'output_cost', 'total_cost'}. Cached input is billed
    at the cheaper cached rate, the remaining (uncached) input at the input
    rate, and output (which already includes reasoning tokens) at the output
    rate. Raises KeyError for a model missing from MODEL_PRICING — cost is never
    silently zeroed or estimated for an unknown model.
    """
    rates = MODEL_PRICING[model]
    uncached_input = input_tokens - cached_input_tokens
    input_cost = (
        uncached_input * rates["input"] + cached_input_tokens * rates["cached_input"]
    ) / 1_000_000
    output_cost = output_tokens * rates["output"] / 1_000_000
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }

# The agent only ever sees the trailing year of categorized history as
# examples: for any transaction, the example pool is restricted to the 12
# calendar months strictly before that transaction's own month. This applies to
# both production runs and the eval harness (agent_eval.py).
ARCHIVE_WINDOW_MONTHS = 12

logger = logging.getLogger(__name__)


@dataclass
class CategorizationResult:
    """Outcome of categorizing a single transaction, with cost/latency.

    On failure, `category` is None and `error` holds a "Type: message" string;
    token counts and `rounds` are whatever had accumulated before the failure.
    """

    category: str | None
    reasoning: str | None
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int  # includes both reasoning and answer tokens
    reasoning_tokens: int
    elapsed_seconds: float
    rounds: int
    tool_invocations: dict = field(default_factory=dict)
    error: str | None = None


def _trailing_window(archive: pd.DataFrame, txn_date) -> pd.DataFrame:
    """Restrict the archive to the ARCHIVE_WINDOW_MONTHS months before txn_date.

    The window is [month_start - N months, month_start): the whole of the
    transaction's own month and everything after it is excluded, so the agent
    never sees same-month or future history. Requires a parsed datetime `Date`
    column on `archive`. If txn_date is missing/unparseable the window is empty
    (the agent then has no examples and leans on its tools) — this is surfaced,
    not silently treated as "all history".
    """
    month_start = pd.Timestamp(txn_date).to_period("M").to_timestamp()
    window_start = month_start - pd.DateOffset(months=ARCHIVE_WINDOW_MONTHS)
    dates = pd.to_datetime(archive["Date"], errors="coerce")
    mask = (dates >= window_start) & (dates < month_start)
    return archive[mask]


def _accumulate_usage(acc: dict, usage) -> None:
    """Add one response's token usage into the running accumulator dict.

    `usage` is present on every successful Responses API result. The nested
    `*_tokens_details` sub-fields (cached / reasoning) legitimately default to 0
    when the API omits them (no cache hit / no reasoning tokens) — that is a
    real zero, not a masked error. A missing top-level `usage` is unexpected and
    is logged rather than silently counted as zero.
    """
    if usage is None:
        logger.warning("response had no usage object; token counts may be undercounted")
        return
    acc["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
    acc["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
    in_details = getattr(usage, "input_tokens_details", None)
    out_details = getattr(usage, "output_tokens_details", None)
    acc["cached_input_tokens"] += getattr(in_details, "cached_tokens", 0) or 0
    acc["reasoning_tokens"] += getattr(out_details, "reasoning_tokens", 0) or 0


SYSTEM_PROMPT_TEMPLATE = """
Your task is to assign a financial transaction into one of the categories listed by the user using the tools available to you.
You are given the 5 most-similar past transactions in the user message.
If they clearly point to one category, categorize_transaction immediately.
If they conflict or are unclear, consult the other tools.
When you feel confident you've gathered all the relevant information, commit the final category using categorize_transaction.

Tips on tool usage:
*search_messages*
    - Default to searching for mentions of the merchant up to 3 months before the transaction
*search_calendar*
    - Default to loading the day of the transaction and the 2 days before
*web_search*
    - Use when the merchant name is unfamiliar and you need to identify what kind of business it is

Personal profile of the user to help disambiguate categories):
{personal_profile}
"""


def _build_priming_message(transaction: dict, similar: pd.DataFrame) -> str:
    if len(similar) == 0:
        similar_block = "  (no similar transactions found in archive)"
    else:
        lines = []
        for i, row in enumerate(similar.itertuples(index=False), start=1):
            lines.append(
                f"  {i}. {row.Name} | ${row.Amount} | {row.Category} "
                f"(similarity={row.Similarity:.2f})"
            )
        similar_block = "\n".join(lines)

    categories_block = "\n".join(f"  - {c}" for c in CATEGORIES)

    return (
        "Transaction:\n"
        f"  Date: {transaction['Date']}\n"
        f"  Name: {transaction['Name']}\n"
        f"  Amount: ${transaction['Amount']}\n"
        f"  Account: {transaction['Account']}\n\n"
        "Five most similar past transactions:\n"
        f"{similar_block}\n\n"
        "Categorize this transaction into exactly ONE of these categories by "
        "calling categorize_transaction:\n"
        f"{categories_block}"
    )


def _validate_category(category: str) -> str:
    if category in CATEGORIES:
        return category
    raise ValueError(f"model returned invalid category: {category!r}")


def _extract_function_calls(output) -> list:
    calls = []
    for item in output:
        if isinstance(item, dict):
            if item.get("type") == "function_call":
                calls.append({
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "call_id": item.get("call_id"),
                })
        else:
            if getattr(item, "type", None) == "function_call":
                calls.append({
                    "name": item.name,
                    "arguments": item.arguments,
                    "call_id": item.call_id,
                })
    return calls


def _serialize_output(output) -> list:
    # Convert SDK objects to plain dicts so they round-trip cleanly as input
    # on the next responses.create call.
    serialized = []
    for item in output:
        if isinstance(item, dict):
            serialized.append(item)
        elif hasattr(item, "model_dump"):
            serialized.append(item.model_dump(exclude_none=True))
        else:
            serialized.append(dict(item))
    return serialized


async def categorize_one(
    transaction: dict,
    archive: pd.DataFrame,
    client: AsyncOpenAI,
    personal_profile: str,
) -> CategorizationResult:
    start = time.perf_counter()
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    rounds = 0
    tool_counts: dict = {}

    def _result(category: str, reasoning: str | None) -> CategorizationResult:
        return CategorizationResult(
            category=_validate_category(category),
            reasoning=reasoning,
            elapsed_seconds=time.perf_counter() - start,
            rounds=rounds,
            tool_invocations=dict(tool_counts),
            **usage,
        )

    windowed = _trailing_window(archive, transaction["Date"])
    similar = top_n_similar(transaction["Name"], windowed, n=5)
    priming = _build_priming_message(transaction, similar)

    input_list = [
        {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(personal_profile=personal_profile)},
        {"role": "user", "content": priming},
    ]

    for _ in range(MAX_ROUNDS):
        rounds += 1
        resp = await client.responses.create(
            model=MODEL,
            input=input_list,
            tools=TOOL_DEFINITIONS,
            tool_choice="required",
            reasoning={"effort": REASONING_EFFORT},
        )
        _accumulate_usage(usage, resp.usage)
        input_list += _serialize_output(resp.output)

        for call in _extract_function_calls(resp.output):
            if call["name"] == "categorize_transaction":
                args = json.loads(call["arguments"])
                return _result(args["category"], args.get("reasoning"))

            tool_counts[call["name"]] = tool_counts.get(call["name"], 0) + 1

            try:
                args = json.loads(call["arguments"] or "{}")
            except json.JSONDecodeError:
                result = json.dumps({"error": "malformed tool arguments JSON"})
            else:
                result = await dispatch_tool(call["name"], args)

            input_list.append({
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": result,
            })

    logger.warning(
        "categorize_one hit MAX_ROUNDS=%d for txn %r; forcing categorize_transaction",
        MAX_ROUNDS, transaction.get("Name"),
    )
    rounds += 1
    final = await client.responses.create(
        model=MODEL,
        input=input_list,
        tools=TOOL_DEFINITIONS,
        tool_choice={"type": "function", "name": "categorize_transaction"},
        reasoning={"effort": REASONING_EFFORT},
    )
    _accumulate_usage(usage, final.usage)
    for call in _extract_function_calls(final.output):
        if call["name"] == "categorize_transaction":
            args = json.loads(call["arguments"])
            return _result(args["category"], args.get("reasoning"))
    raise RuntimeError(
        f"forced categorize_transaction returned no function_call for txn {transaction!r}"
    )


async def categorize_dataframe(
    df: pd.DataFrame,
    archive: pd.DataFrame,
    client: AsyncOpenAI,
    personal_profile: str,
    concurrency: int = 5,
) -> list[CategorizationResult]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(txn: dict) -> CategorizationResult:
        async with sem:
            try:
                return await categorize_one(txn, archive, client, personal_profile)
            except Exception as e:
                logger.exception("categorize_one failed for %r", txn.get("Name"))
                return CategorizationResult(
                    category=None,
                    reasoning=None,
                    input_tokens=0,
                    cached_input_tokens=0,
                    output_tokens=0,
                    reasoning_tokens=0,
                    elapsed_seconds=0.0,
                    rounds=0,
                    error=f"{type(e).__name__}: {e}",
                )

    records = df.to_dict(orient="records")
    return await asyncio.gather(*(bounded(r) for r in records))
