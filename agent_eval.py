"""Held-out evaluation harness for the transaction-categorizer agent.

Deterministically holds out a fraction (default 25%) of the archived,
ground-truth-categorized transactions from the first K months of a year, then
re-categorizes them with the agent while hiding the held-out rows from its
example pool. Writes a single CSV report with token cost, runtime, overall
accuracy, per-category sensitivity/precision, and a per-transaction breakdown.

The example pool the agent sees is governed by agent.categorize_one's trailing
12-month window (ARCHIVE_WINDOW_MONTHS): for a held-out transaction in month m,
the agent only sees archived transactions from the 12 months strictly before m,
minus every held-out row. This requires the prior-year archive sheet
(PAST_YEAR_ARCHIVE_SHEET_ID) so that early-year transactions still have history.

Run: .venv/bin/python agent_eval.py [--k 4] [--year 2026] [--frac 0.25]
                                     [--seed 42] [--concurrency 5]
                                     [--output agent_eval_report.csv]

NOTE: like categorize_transactions.py, this makes live OpenAI Responses calls
AND live tool calls (iMessage / Google Calendar / Perplexity) for every held-out
transaction, so it needs the same setup (.env, OAuth, Full Disk Access).
"""

import argparse
import asyncio
import csv
import statistics
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

from agent import categorize_dataframe, compute_cost, MODEL
from categorize_transactions import load_archive
from tools import CATEGORIES, TOOL_DEFINITIONS

# Non-terminal tools the agent may consult (categorize_transaction is the
# terminal commit, not a consultation, so it is excluded from usage stats).
EVAL_TOOLS = [t["name"] for t in TOOL_DEFINITIONS if t["name"] != "categorize_transaction"]


DEFAULT_K = 4
DEFAULT_YEAR = datetime.now().year
DEFAULT_FRAC = 0.25
DEFAULT_SEED = 42
DEFAULT_CONCURRENCY = 5
DEFAULT_OUTPUT = "agent_eval_report.csv"


def select_held_out(archive, year, k, frac, seed):
    """Deterministically pick `frac` of each of months 1..k of `year`.

    Returns (held_out_df, n_candidates). Determinism comes from a stable
    pre-sort plus a fixed random_state, so the same rows are chosen every run
    regardless of the order the sheet returns them in.
    """
    candidates = archive[
        (archive["Date"].dt.year == year)
        & (archive["Date"].dt.month.between(1, k))
    ].copy()
    if candidates.empty:
        return candidates, 0

    candidates = candidates.sort_values(
        ["Date", "Name", "Amount", "Account"], kind="mergesort"
    )
    held_out = candidates.groupby(
        candidates["Date"].dt.month, group_keys=False
    ).sample(frac=frac, random_state=seed)
    return held_out, len(candidates)


def _safe_ratio(numerator, denominator):
    # Returns "N/A" rather than 0 or a ZeroDivisionError when the denominator is
    # 0, so an undefined metric is never mistaken for a real 0.0 score.
    if denominator == 0:
        return "N/A"
    return round(numerator / denominator, 4)


def per_category_metrics(rows):
    """rows: list of (true_category, predicted_category_or_None)."""
    metrics = []
    for c in CATEGORIES:
        tp = sum(1 for t, p in rows if t == c and p == c)
        fp = sum(1 for t, p in rows if p == c and t != c)
        fn = sum(1 for t, p in rows if t == c and p != c)
        support = tp + fn
        # Skip categories that never appear as either truth or prediction —
        # they would just be noise rows of N/A.
        if support == 0 and fp == 0:
            continue
        metrics.append({
            "category": c,
            "support": support,
            "sensitivity": _safe_ratio(tp, tp + fn),
            "precision": _safe_ratio(tp, tp + fp),
        })
    return metrics


def tool_usage_stats(results):
    """For each non-terminal tool: how many runs invoked it (and how often)."""
    n = len(results)
    stats = []
    for tool in EVAL_TOOLS:
        runs_invoked = sum(1 for r in results if r.tool_invocations.get(tool, 0) > 0)
        total = sum(r.tool_invocations.get(tool, 0) for r in results)
        stats.append({
            "tool": tool,
            "runs_invoked": runs_invoked,
            "pct_runs": _safe_ratio(runs_invoked, n),
            "total_invocations": total,
        })
    return stats


def write_report(path, *, config, summary, tool_usage, category_metrics, detail_rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)

        w.writerow(["## Run summary"])
        w.writerow(["metric", "value"])
        for key, value in {**config, **summary}.items():
            w.writerow([key, value])

        w.writerow([])
        w.writerow(["## Tool usage"])
        w.writerow(["tool", "runs_invoked", "pct_runs", "total_invocations"])
        for t in tool_usage:
            w.writerow([t["tool"], t["runs_invoked"], t["pct_runs"], t["total_invocations"]])

        w.writerow([])
        w.writerow(["## Per-category metrics"])
        w.writerow(["category", "support", "sensitivity", "precision"])
        for m in category_metrics:
            w.writerow([m["category"], m["support"], m["sensitivity"], m["precision"]])

        w.writerow([])
        w.writerow(["## Held-out transactions"])
        w.writerow([
            "Date", "Name", "Amount", "Account",
            "TrueCategory", "AssignedCategory", "Correct", "Rationale",
        ])
        for r in detail_rows:
            w.writerow(r)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Number of leading months of the year to sample from.")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR,
                        help="Calendar year whose first k months are evaluated.")
    parser.add_argument("--frac", type=float, default=DEFAULT_FRAC,
                        help="Fraction of each month's transactions to hold out.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for deterministic held-out selection.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    archive = load_archive()

    # Rows with an unparseable/blank Date can't be placed in a month, so they
    # can be neither held out nor windowed. Drop them outright and report how
    # many — they are not silently mixed into the pool.
    n_dropped_no_date = int(archive["Date"].isna().sum())
    archive = archive[archive["Date"].notna()].reset_index(drop=True)

    held_out, n_candidates = select_held_out(
        archive, args.year, args.k, args.frac, args.seed
    )
    if held_out.empty:
        raise SystemExit(
            f"No archived transactions found for months 1..{args.k} of "
            f"{args.year}. Nothing to evaluate."
        )

    # The agent must never see a held-out row as an example. Removing all of
    # them globally (rather than per-transaction) also stops one held-out row
    # from leaking into another's example pool. The trailing-12-month window is
    # then applied per transaction inside agent.categorize_one.
    pool_archive = archive.drop(held_out.index)

    print(
        f"Evaluating {len(held_out)} held-out transactions "
        f"(of {n_candidates} candidates in months 1..{args.k} of {args.year}); "
        f"example pool = {len(pool_archive)} archived rows, trailing 12 months each."
    )

    client = AsyncOpenAI()
    results = await categorize_dataframe(
        held_out, pool_archive, client,
        concurrency=args.concurrency,
    )

    truths = held_out["Category"].tolist()
    preds = [r.category for r in results]  # None on error

    n_total = len(results)
    n_correct = sum(1 for t, p in zip(truths, preds) if p is not None and t == p)
    n_errors = sum(1 for r in results if r.error is not None)

    # Token totals across every agent invocation.
    total_input = sum(r.input_tokens for r in results)
    cached_input = sum(r.cached_input_tokens for r in results)
    total_output = sum(r.output_tokens for r in results)
    reasoning_output = sum(r.reasoning_tokens for r in results)

    # Runtime stats over successful invocations only — errored ones recorded 0s
    # and would drag the distribution toward zero.
    runtimes = [r.elapsed_seconds for r in results if r.error is None]

    def _stat(fn):
        return round(fn(runtimes), 3) if runtimes else "N/A"

    cost = compute_cost(MODEL, total_input, cached_input, total_output)

    config = {
        "model": MODEL,
        "target_year": args.year,
        "k_months": args.k,
        "holdout_fraction": args.frac,
        "seed": args.seed,
        "concurrency": args.concurrency,
    }
    summary = {
        "n_candidates": n_candidates,
        "n_held_out": n_total,
        "n_errors": n_errors,
        "n_dropped_no_date": n_dropped_no_date,
        "overall_accuracy": round(n_correct / n_total, 4) if n_total else "N/A",
        "total_input_tokens": total_input,
        "cached_input_tokens": cached_input,
        "uncached_input_tokens": total_input - cached_input,
        "total_output_tokens": total_output,
        "reasoning_output_tokens": reasoning_output,
        "answer_output_tokens": total_output - reasoning_output,
        "input_cost_usd": round(cost["input_cost"], 6),
        "output_cost_usd": round(cost["output_cost"], 6),
        "total_cost_usd": round(cost["total_cost"], 6),
        "mean_runtime_seconds": _stat(statistics.mean),
        "median_runtime_seconds": _stat(statistics.median),
        "max_runtime_seconds": _stat(max),
    }

    category_metrics = per_category_metrics(list(zip(truths, preds)))

    detail_rows = []
    for (_, txn), r in zip(held_out.iterrows(), results):
        pred = r.category
        correct = "" if pred is None else str(txn["Category"] == pred)
        rationale = r.reasoning if r.error is None else f"ERROR: {r.error}"
        detail_rows.append([
            txn["Date"].date().isoformat(),
            txn["Name"],
            txn["Amount"],
            txn["Account"],
            txn["Category"],
            pred if pred is not None else "",
            correct,
            rationale or "",
        ])

    write_report(
        args.output,
        config=config,
        summary=summary,
        tool_usage=tool_usage_stats(results),
        category_metrics=category_metrics,
        detail_rows=detail_rows,
    )

    print(f"\nAccuracy: {summary['overall_accuracy']} over {n_total} transactions "
          f"({n_errors} errored).")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
