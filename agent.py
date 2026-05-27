import asyncio
import json
import logging

import pandas as pd
from openai import AsyncOpenAI

from archive_index import top_n_similar
from tools import CATEGORIES, TOOL_DEFINITIONS, dispatch_tool


MODEL = "gpt-5.4-mini"
MAX_ROUNDS = 10

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
Your task is to assign a financial transaction into one of the categories listed by the user using the tools available to you.
You are given the 5 most-similar past transactions in the user message.
If they clearly point to one category, categorize_transaction immediately.
If they conflict or are unclear, consult the other tools.
If a tool response is truncated, narrow the date range or use a more specific keyword.
When you feel confident you've gathered all the relevant information, commit the final category using categorize_transaction.

Tips on tool usage:
*search_messages*
    - Default to searching for mentions of the merchant up to 3 months before the transaction
*search_calendar*
    - Default to loading the day of the transaction and the 2 days before
*web_search*
    - Use when the merchant name is unfamiliar and you need to identify what kind of business it is

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
        f"  {', '.join(CATEGORIES)}"
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
) -> str:
    similar = top_n_similar(transaction["Name"], archive, n=5)
    priming = _build_priming_message(transaction, similar)

    input_list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": priming},
    ]

    for _ in range(MAX_ROUNDS):
        resp = await client.responses.create(
            model=MODEL,
            input=input_list,
            tools=TOOL_DEFINITIONS,
            tool_choice="required",
        )
        input_list += _serialize_output(resp.output)

        for call in _extract_function_calls(resp.output):
            if call["name"] == "categorize_transaction":
                args = json.loads(call["arguments"])
                return _validate_category(args["category"])

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
    final = await client.responses.create(
        model=MODEL,
        input=input_list,
        tools=TOOL_DEFINITIONS,
        tool_choice={"type": "function", "name": "categorize_transaction"},
    )
    for call in _extract_function_calls(final.output):
        if call["name"] == "categorize_transaction":
            args = json.loads(call["arguments"])
            return _validate_category(args["category"])
    raise RuntimeError(
        f"forced categorize_transaction returned no function_call for txn {transaction!r}"
    )


async def categorize_dataframe(
    df: pd.DataFrame,
    archive: pd.DataFrame,
    client: AsyncOpenAI,
    concurrency: int = 5,
) -> list:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(txn: dict) -> str:
        async with sem:
            try:
                return await categorize_one(txn, archive, client)
            except Exception as e:
                logger.exception("categorize_one failed for %r", txn.get("Name"))
                return f"ERROR: {type(e).__name__}: {e}"

    records = df.to_dict(orient="records")
    return await asyncio.gather(*(bounded(r) for r in records))
