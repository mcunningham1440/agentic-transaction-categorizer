"""Held-out evaluation harness for the transaction-categorizer agent.

Deterministically holds out a fraction (default 25%) of the archived,
ground-truth-categorized transactions from the first K months of a year, then
re-categorizes them with the agent while hiding the held-out rows from its
example pool.

Every eval now runs through LangSmith: the held-out rows are synced to a
LangSmith **dataset** and the agent is run against them as an **experiment** via
`aevaluate`, scored by an exact-match `correct` evaluator. Results (per-example
traces + auto token/cost/latency) land in LangSmith's Datasets & Experiments
tab. The local CSV report that the old `agent_eval` produced (token cost,
runtime, per-category sensitivity/precision, per-tool usage, per-transaction
breakdown) is now **optional**, written only when `--csv` is passed.

The example pool the agent sees is governed by agent.categorize_one's trailing
12-month window (ARCHIVE_WINDOW_MONTHS): for a held-out transaction in month m,
the agent only sees archived transactions from the 12 months strictly before m,
minus every held-out row. This requires the prior-year archive sheet
(PAST_YEAR_ARCHIVE_SHEET_ID) so that early-year transactions still have history.

Run: .venv/bin/python agent_eval.py [--k 4] [--year 2026] [--frac 0.25]
                                     [--seed 42] [--concurrency 5]
                                     [--csv [PATH]]

`--csv` with no path writes to output/agent_eval_report.csv; `--csv PATH`
overrides the destination. Omit it to run the experiment only.

NOTE: like categorize_transactions.py, this makes live OpenAI Responses calls
AND live tool calls (iMessage / Google Calendar / Perplexity) for every held-out
transaction, so it needs the same setup (.env, OAuth, Full Disk Access). It also
needs LANGSMITH_API_KEY for the dataset/experiment sync.
"""

import argparse
import asyncio
import csv
import os
import statistics
from datetime import datetime

import pandas as pd
from langsmith import Client

from categorizer.agent import categorize_one, compute_cost, MODEL
from categorizer.categories import CATEGORIES
from categorizer.paths import OUTPUT_DIR
from categorizer.pipeline import load_archive
from categorizer.tools import TOOL_NAMES

# Non-terminal tools the agent may consult. The terminal commit is now
# structured output rather than a tool, so TOOL_NAMES already excludes it.
EVAL_TOOLS = list(TOOL_NAMES)


DEFAULT_K = 4
DEFAULT_YEAR = datetime.now().year
DEFAULT_FRAC = 0.25
DEFAULT_SEED = 42
DEFAULT_CONCURRENCY = 5
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "agent_eval_report.csv")


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


def _assign_example_ids(held_out):
    """Attach a stable, content-derived `example_id` to each held-out row.

    The id is `date|name|amount|account|occurrence`, where `occurrence` is a
    per-(date,name,amount,account) counter assigned in a deterministic sort
    order. This survives dataset reuse (same --k/--year/--frac/--seed selects
    the same rows and so regenerates identical ids) and disambiguates genuine
    duplicate transactions, so predictions can be joined back to ground truth
    for the CSV without relying on DataFrame index labels or row ordering.
    """
    ho = held_out.sort_values(
        ["Date", "Name", "Amount", "Account"], kind="mergesort"
    ).copy()
    ids = []
    seen: dict = {}
    for row in ho.itertuples(index=False):
        key = (row.Date, row.Name, row.Amount, row.Account)
        n = seen.get(key, 0)
        seen[key] = n + 1
        ids.append(
            f"{row.Date.date().isoformat()}|{row.Name}|{row.Amount}|{row.Account}|{n}"
        )
    ho["example_id"] = ids
    return ho


def _example_payloads(held_out):
    """One LangSmith example per held-out transaction.

    `inputs` mirrors the dict the agent consumes (Date stored as an ISO string
    and re-parsed in the target) plus the `example_id` used to join predictions
    back to ground truth for the CSV — categorize_one ignores unknown input
    keys, so `example_id` never reaches the agent. `outputs` is the
    reference/ground-truth category the `correct` evaluator grades against.
    """
    examples = []
    for row in held_out.itertuples(index=False):
        examples.append({
            "inputs": {
                "Date": row.Date.date().isoformat(),
                "Name": row.Name,
                "Amount": float(row.Amount),
                "Account": row.Account,
                "example_id": row.example_id,
            },
            "outputs": {"category": row.Category},
        })
    return examples


def sync_dataset(client, dataset_name, held_out):
    """Create the dataset + examples once; reuse it as-is on later runs.

    NOTE: if a dataset with this name already exists it is reused unchanged, not
    refreshed. The name encodes year/k/frac/seed and selection is deterministic,
    so the examples are identical across reruns of the same config — but if you
    change selection logic without changing those args, the stale dataset
    persists. Delete it in the UI (or bump the name) to force a rebuild.
    """
    if client.has_dataset(dataset_name=dataset_name):
        return client.read_dataset(dataset_name=dataset_name)
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Held-out archived transactions for categorizer agent eval.",
    )
    client.create_examples(dataset_id=dataset.id, examples=_example_payloads(held_out))
    return dataset


def _build_target(pool_archive, sink):
    """Build the async target the experiment runs for each dataset example.

    Closes over `pool_archive` (the archive minus all held-out rows) so the
    agent's trailing-window/similarity lookup never sees a held-out row, and
    over `sink`, a dict the target populates with the full per-example
    CategorizationResult fields (tokens, tool counts, runtime) keyed by
    `example_id` — that is where the optional CSV report gets its data, since
    the returned `outputs` only needs to carry what the `correct` evaluator
    grades.
    """
    async def target(inputs: dict) -> dict:
        txn = {
            "Date": pd.Timestamp(inputs["Date"]),
            "Name": inputs["Name"],
            "Amount": inputs["Amount"],
            "Account": inputs["Account"],
        }
        eid = inputs.get("example_id")
        try:
            r = await categorize_one(txn, pool_archive)
            record = {
                "predicted_category": r.category,
                "reasoning": r.reasoning,
                "error": r.error,
                "input_tokens": r.input_tokens,
                "cached_input_tokens": r.cached_input_tokens,
                "output_tokens": r.output_tokens,
                "reasoning_tokens": r.reasoning_tokens,
                "elapsed_seconds": r.elapsed_seconds,
                "tool_invocations": dict(r.tool_invocations),
            }
        except Exception as e:  # noqa: BLE001
            # categorize_one *raises* on the recursion limit / missing
            # structured_response (agent.py). If that propagated out of the
            # target, LangSmith would mark the run errored and skip the
            # `correct` evaluator entirely — the example would show "No
            # feedback" and silently drop out of accuracy. Mirror the old
            # bounded() wrapper instead: score it as a None category (False) and
            # zero its token/runtime counts so the example stays in the
            # denominator without corrupting the aggregates.
            record = {
                "predicted_category": None,
                "reasoning": None,
                "error": f"{type(e).__name__}: {e}",
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "elapsed_seconds": 0.0,
                "tool_invocations": {},
            }
        if eid is not None:
            sink[eid] = record
        return {
            "category": record["predicted_category"],
            "reasoning": record["reasoning"],
            "error": record["error"],
        }

    return target


def correct(outputs: dict, reference_outputs: dict) -> bool:
    """Exact match: did the agent assign the archived ground-truth category?

    A None category (the agent errored) is scored False rather than skipped, so
    errors count against accuracy instead of silently leaving the denominator.
    """
    predicted = outputs.get("category")
    return predicted is not None and predicted == reference_outputs.get("category")


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


def tool_usage_stats(tool_invocation_dicts):
    """For each non-terminal tool: how many runs invoked it (and how often).

    Takes the per-run `tool_invocations` mappings ({tool_name: count}).
    """
    n = len(tool_invocation_dicts)
    stats = []
    for tool in EVAL_TOOLS:
        runs_invoked = sum(1 for ti in tool_invocation_dicts if ti.get(tool, 0) > 0)
        total = sum(ti.get(tool, 0) for ti in tool_invocation_dicts)
        stats.append({
            "tool": tool,
            "runs_invoked": runs_invoked,
            "pct_runs": _safe_ratio(runs_invoked, n),
            "total_invocations": total,
        })
    return stats


def write_report(path, *, config, summary, tool_usage, category_metrics, detail_rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
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


def write_csv_report(path, *, config, n_candidates, n_dropped_no_date, truth_map, sink):
    """Assemble the local CSV report from the experiment's collected results.

    `truth_map` is {example_id: held_out row}; `sink` is {example_id: record}
    populated by the target. An example_id present in truth_map but missing from
    sink (target never ran or the dataset predates example_id tagging) is
    surfaced as an explicit error row rather than dropped, so the CSV's
    denominator always matches the held-out count.
    """
    missing_record = {
        "predicted_category": None,
        "reasoning": None,
        "error": "no result recorded for this example",
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "elapsed_seconds": 0.0,
        "tool_invocations": {},
    }

    truths, preds, records, detail_rows = [], [], [], []
    for eid, txn in truth_map.items():
        rec = sink.get(eid, missing_record)
        pred = rec["predicted_category"]
        truths.append(txn.Category)
        preds.append(pred)
        records.append(rec)

        correct_str = "" if pred is None else str(txn.Category == pred)
        rationale = rec["reasoning"] if rec["error"] is None else f"ERROR: {rec['error']}"
        detail_rows.append([
            txn.Date.date().isoformat(),
            txn.Name,
            txn.Amount,
            txn.Account,
            txn.Category,
            pred if pred is not None else "",
            correct_str,
            rationale or "",
        ])

    n_total = len(records)
    n_correct = sum(1 for t, p in zip(truths, preds) if p is not None and t == p)
    n_errors = sum(1 for r in records if r["error"] is not None)

    total_input = sum(r["input_tokens"] for r in records)
    cached_input = sum(r["cached_input_tokens"] for r in records)
    total_output = sum(r["output_tokens"] for r in records)
    reasoning_output = sum(r["reasoning_tokens"] for r in records)

    # Runtime stats over successful invocations only — errored ones recorded 0s
    # and would drag the distribution toward zero.
    runtimes = [r["elapsed_seconds"] for r in records if r["error"] is None]

    def _stat(fn):
        return round(fn(runtimes), 3) if runtimes else "N/A"

    cost = compute_cost(MODEL, total_input, cached_input, total_output)

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

    write_report(
        path,
        config=config,
        summary=summary,
        tool_usage=tool_usage_stats([r["tool_invocations"] for r in records]),
        category_metrics=per_category_metrics(list(zip(truths, preds))),
        detail_rows=detail_rows,
    )
    return summary


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
    parser.add_argument(
        "--csv", nargs="?", const=DEFAULT_OUTPUT, default=None, metavar="PATH",
        help="Also write the local CSV report. Bare --csv uses the default "
             f"path ({DEFAULT_OUTPUT}); --csv PATH overrides it. Omit to run "
             "the LangSmith experiment only.",
    )
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
    ho = _assign_example_ids(held_out)
    truth_map = {row.example_id: row for row in ho.itertuples(index=False)}

    client = Client()
    dataset_name = (
        f"txn-categorizer-{args.year}-k{args.k}-frac{args.frac}-seed{args.seed}"
    )
    sync_dataset(client, dataset_name, ho)

    print(
        f"Dataset '{dataset_name}': {len(ho)} held-out examples "
        f"(of {n_candidates} candidates in months 1..{args.k} of {args.year}); "
        f"example pool = {len(pool_archive)} rows. Running experiment..."
    )

    sink: dict = {}
    results = await client.aevaluate(
        _build_target(pool_archive, sink),
        data=dataset_name,
        evaluators=[correct],
        max_concurrency=args.concurrency,
        experiment_prefix=f"{MODEL}-k{args.k}-frac{args.frac}",
        metadata={
            "model": MODEL,
            "k": args.k,
            "year": args.year,
            "frac": args.frac,
            "seed": args.seed,
        },
    )

    print(f"\nExperiment '{results.experiment_name}' complete.")
    try:
        print(f"View it here: {results.url}")
    except Exception:
        # `url` is a convenience accessor; if the SDK can't build it, fall back
        # to a pointer rather than failing the whole run after the eval is done.
        print(f"Find it under Datasets & Experiments → '{dataset_name}' in LangSmith.")

    if args.csv is not None:
        config = {
            "model": MODEL,
            "target_year": args.year,
            "k_months": args.k,
            "holdout_fraction": args.frac,
            "seed": args.seed,
            "concurrency": args.concurrency,
        }
        summary = write_csv_report(
            args.csv,
            config=config,
            n_candidates=n_candidates,
            n_dropped_no_date=n_dropped_no_date,
            truth_map=truth_map,
            sink=sink,
        )
        print(
            f"\nAccuracy: {summary['overall_accuracy']} over {summary['n_held_out']} "
            f"transactions ({summary['n_errors']} errored)."
        )
        print(f"CSV report written to {args.csv}")


if __name__ == "__main__":
    asyncio.run(main())
