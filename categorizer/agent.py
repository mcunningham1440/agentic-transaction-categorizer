import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd
from langchain.agents import create_agent
from langchain_fireworks import ChatFireworks
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from openai import AsyncOpenAI
from pydantic import Field, create_model

from categorizer.archive.index import top_n_similar
from categorizer.categories import CATEGORIES, CATEGORY_INSTRUCTIONS
from categorizer.tools import AGENT_TOOLS


# Which hosted API serves MODEL. "openai" reads OPENAI_API_KEY, "fireworks"
# reads FIREWORKS_API_KEY. Fireworks model ids are fully qualified, e.g.
# "accounts/fireworks/models/deepseek-v4-flash".
PROVIDER: Literal["openai", "fireworks"] = "openai"
MODEL = "gpt-5.6-luna"
# Set to None for models that don't accept a reasoning_effort request field
# (most non-reasoning open-weight models on Fireworks reject it outright).
REASONING_EFFORT: str | None = "low"
MAX_ROUNDS = 10

# USD per 1M tokens. Keyed by model id; OpenAI and Fireworks ids never collide.
# Fireworks rates are the Standard serverless tier (docs.fireworks.ai/serverless/pricing,
# checked 2026-08-02); the Fast/Priority tiers cost more, so a run pinned to one
# of those tiers is under-costed by this table.
MODEL_PRICING = {
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.6-luna": {"input": 1.00, "cached_input": 0.10, "output": 6.00},
    "accounts/fireworks/models/kimi-k3": {"input": 3.00, "cached_input": 0.30, "output": 15.00},
    "accounts/fireworks/models/deepseek-v4-pro": {"input": 1.74, "cached_input": 0.145, "output": 3.48},
    "accounts/fireworks/models/deepseek-v4-flash": {"input": 0.14, "cached_input": 0.028, "output": 0.28},
    "accounts/fireworks/models/glm-5p2": {"input": 1.40, "cached_input": 0.14, "output": 4.40},
    "accounts/fireworks/models/qwen3p7-plus": {"input": 0.40, "cached_input": 0.08, "output": 1.60},
    "accounts/fireworks/models/minimax-m3": {"input": 0.30, "cached_input": 0.06, "output": 1.20},
    "accounts/fireworks/models/gpt-oss-120b": {"input": 0.15, "cached_input": 0.015, "output": 0.60},
    "accounts/fireworks/models/gpt-oss-20b": {"input": 0.07, "cached_input": 0.035, "output": 0.30},
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


def _aggregate_run(messages) -> tuple[dict, dict, int]:
    """Sum token usage and tool invocations across one agent run's messages.

    LangChain attaches per-call token counts to each AIMessage's
    `usage_metadata` (input/output totals plus `input_token_details.cache_read`
    and `output_token_details.reasoning`). Summing over every AIMessage captures
    all model calls in the loop, including the final structured-output call.

    `rounds` is the number of model calls (AIMessages carrying usage_metadata),
    the LangChain analog of the old per-request round count.

    Note: the nested detail fields default to 0 via `.get(..., 0)` when the
    provider omits them (no cache hit / no reasoning tokens) — a real zero, not a
    masked error; this matches the previous Responses-API behavior. These counts
    feed compute_cost(), so an omitted detail field undercounts rather than
    erroring.
    """
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    tool_counts: dict = {}
    rounds = 0
    for m in messages:
        um = getattr(m, "usage_metadata", None)
        if um:
            rounds += 1
            usage["input_tokens"] += um.get("input_tokens", 0) or 0
            usage["output_tokens"] += um.get("output_tokens", 0) or 0
            usage["cached_input_tokens"] += (um.get("input_token_details") or {}).get("cache_read", 0) or 0
            usage["reasoning_tokens"] += (um.get("output_token_details") or {}).get("reasoning", 0) or 0
        for call in getattr(m, "tool_calls", None) or []:
            name = call.get("name")
            tool_counts[name] = tool_counts.get(name, 0) + 1
    return usage, tool_counts, rounds


def _build_categories_block() -> str:
    """Render the authoritative category list (with per-category instructions).

    This is identical for every transaction, so it lives in the static system
    prompt rather than the per-transaction user message — that keeps it inside
    the cacheable prompt prefix (see SYSTEM_PROMPT / _build_priming_message).
    """
    cat_lines = []
    for c in CATEGORIES:
        instr = CATEGORY_INSTRUCTIONS.get(c, "")
        if instr:
            # Indent any wrapped/multi-line instructions under the bullet.
            cat_lines.append(f"  - {c}: {instr.replace(chr(10), chr(10) + '      ')}")
        else:
            cat_lines.append(f"  - {c}")
    return "\n".join(cat_lines)

SYSTEM_PROMPT = """
Your task is to assign a financial transaction into one of the categories listed below using the tools available to you.
You are given the 5 most-similar past transactions in the user message.
If they clearly point to one category, give your final answer immediately.
If they conflict or are unclear, consult the other tools.
When you feel confident you've gathered all the relevant information, respond with the final category and a brief justification.

Tips on tool usage:
*search_messages*
    - Default to searching for mentions of the merchant up to 3 months before the transaction
*search_calendar*
    - Default to loading the day of the transaction and the 2 days before
*web_search*
    - Use when the merchant name is unfamiliar and you need to identify what kind of business it is
*search_gmail*
    - Always use to identify items in E-commerce orders (e.g. Amazon) and Zelle transactions
    - Try searching vendor AND amount, e.g. "Amazon 25.21"
    - Use a date range +/- 2 days from the transaction and expand up to +/- 2 weeks if necessary

Some categories carry their own handling instructions; these appear after the
category name in the list below. Follow them.

Assign the transaction to exactly ONE of these categories:
""" + _build_categories_block()


# Structured-output schema the agent commits its answer to (replaces the old
# terminal `categorize_transaction` tool). The category is constrained to the
# authoritative CATEGORIES at import; `_validate_category` re-checks it as a
# belt-and-suspenders guard.
Categorization = create_model(
    "Categorization",
    category=(
        Literal[tuple(CATEGORIES)],
        Field(description="The chosen category. Must be one of the listed values"),
    ),
    reasoning=(str, Field(description="Brief justification for the chosen category")),
    __doc__="The committed category for the transaction and a brief justification.",
)


# Stable cache key shared by every request so the provider routes them to the
# same machine for prefix-cache hits; bump the version suffix when the static
# prefix (system prompt, tools, schema, category list) changes materially.
PROMPT_CACHE_KEY = "txn-categorizer/v1"


def _build_model():
    """Build the chat model for PROVIDER/MODEL.

    OpenAI runs on the Responses API (use_responses_api=True) to preserve
    reasoning-effort behavior, with explicit prompt-cache routing/retention.

    Fireworks is served by langchain-fireworks' ChatFireworks (Chat Completions
    under the hood). Two provider differences worth knowing, neither of which
    errors out:
      - Prompt caching is automatic and has no cache-key parameter; the OpenAI
        `prompt_cache_key`/`prompt_cache_retention` fields do not exist there.
        The `user` field is Fireworks' documented session-affinity hint, so it
        carries PROMPT_CACHE_KEY to pin related requests to one replica. If
        Fireworks does not report `prompt_tokens_details.cached_tokens`,
        `cached_input_tokens` stays 0 and reported cost is an *over*estimate.
      - langchain-fireworks does not map reasoning tokens into
        `output_token_details`, so `reasoning_tokens` is reported as 0 even for
        reasoning models. `output_tokens` still includes them, so cost is
        unaffected — only the reasoning-token column in the eval report is.

    Raises ValueError for an unknown PROVIDER rather than defaulting to OpenAI.
    """
    if PROVIDER == "openai":
        return ChatOpenAI(
            model=MODEL,
            use_responses_api=True,
            reasoning_effort=REASONING_EFFORT,
            model_kwargs={
                "prompt_cache_key": PROMPT_CACHE_KEY,
                "prompt_cache_retention": "24h",
            },
        )
    if PROVIDER == "fireworks":
        return ChatFireworks(
            model=MODEL,
            reasoning_effort=REASONING_EFFORT,
            model_kwargs={"user": PROMPT_CACHE_KEY},
        )
    raise ValueError(f"unknown PROVIDER: {PROVIDER!r}")


# Built once at import and reused across all transactions; create_agent runs the
# model+tools loop and emits structured output.
_model = _build_model()
agent = create_agent(
    model=_model,
    tools=AGENT_TOOLS,
    system_prompt=SYSTEM_PROMPT,
    response_format=Categorization,
)


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
        f"{similar_block}"
    )


def _validate_category(category: str) -> str:
    if category in CATEGORIES:
        return category
    raise ValueError(f"model returned invalid category: {category!r}")


async def categorize_one(
    transaction: dict,
    archive: pd.DataFrame,
    client: AsyncOpenAI | None = None,
) -> CategorizationResult:
    # `client` is accepted but unused: the LangChain model and agent are built
    # once at module load. It is kept in the signature so existing callers
    # (pipeline.py, eval.py, categorize_dataframe) need no changes.
    start = time.perf_counter()

    windowed = _trailing_window(archive, transaction["Date"])
    similar = top_n_similar(transaction["Name"], windowed, n=5)
    priming = _build_priming_message(transaction, similar)

    try:
        # Each agent round is ~2 supersteps (model call + tool call); the final
        # structured-output step adds a couple more. Map MAX_ROUNDS to a
        # recursion limit with a small buffer for the structured-output nodes.
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": priming}]},
            config={"recursion_limit": MAX_ROUNDS * 2 + 3},
        )
    except GraphRecursionError as e:
        # Hit the loop ceiling without committing a category. The old loop
        # force-called the terminal tool here; with structured output there is
        # no clean force hook, so surface this as an error (caught by
        # categorize_dataframe's bounded wrapper into an error row) rather than
        # fabricating a best guess.
        raise RuntimeError(
            f"agent hit recursion limit without a category for txn "
            f"{transaction.get('Name')!r}"
        ) from e

    usage, tool_counts, rounds = _aggregate_run(result["messages"])

    structured = result.get("structured_response")
    if structured is None:
        raise RuntimeError(
            f"agent produced no structured_response for txn {transaction.get('Name')!r}"
        )

    return CategorizationResult(
        category=_validate_category(structured.category),
        reasoning=structured.reasoning,
        elapsed_seconds=time.perf_counter() - start,
        rounds=rounds,
        tool_invocations=tool_counts,
        **usage,
    )


async def categorize_dataframe(
    df: pd.DataFrame,
    archive: pd.DataFrame,
    client: AsyncOpenAI | None = None,
    concurrency: int = 20,
) -> list[CategorizationResult]:
    sem = asyncio.Semaphore(concurrency)

    async def bounded(txn: dict) -> CategorizationResult:
        async with sem:
            try:
                return await categorize_one(txn, archive, client)
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
