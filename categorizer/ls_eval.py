"""LangSmith Datasets + Experiments harness for the categorizer agent.

Companion to agent_eval.py. It reuses the exact same deterministic held-out
selection (categorizer.eval.select_held_out), but instead of computing metrics
locally and writing a CSV it:

  1. Syncs the held-out transactions to a LangSmith **dataset** (inputs = the
     transaction fields, reference output = the archived ground-truth category).
  2. Runs the agent against that dataset as a LangSmith **experiment** via
     `aevaluate`, scoring each prediction with a `correct` exact-match evaluator.

Results then appear under LangSmith's *Datasets & Experiments* tab, with one
nested agent trace per example and automatic token/cost/latency aggregation
(the per-call traces come from the same create_agent instrumentation that
agent_eval.py relies on for tracing). Accuracy here should match agent_eval.py
for the same --k/--year/--frac/--seed, since both hold out identical rows.

Run: .venv/bin/python ls_eval.py [--k 4] [--year 2026] [--frac 0.25]
                                 [--seed 42] [--concurrency 5]

Requires LANGSMITH_API_KEY (+ LANGSMITH_TRACING) in config/.env, plus the same
live setup as agent_eval.py (OpenAI, Google OAuth, Full Disk Access).
"""

import argparse
import asyncio

import pandas as pd
from langsmith import Client

from categorizer.agent import MODEL, categorize_one
from categorizer.eval import (
    DEFAULT_CONCURRENCY,
    DEFAULT_FRAC,
    DEFAULT_K,
    DEFAULT_SEED,
    DEFAULT_YEAR,
    select_held_out,
)
from categorizer.pipeline import load_archive


def _example_payloads(held_out):
    """One LangSmith example per held-out transaction.

    `inputs` mirrors the dict the agent consumes (Date stored as an ISO string
    and re-parsed in the target). `outputs` is the reference/ground-truth
    category the `correct` evaluator grades against.
    """
    examples = []
    for _, txn in held_out.iterrows():
        examples.append({
            "inputs": {
                "Date": txn["Date"].date().isoformat(),
                "Name": txn["Name"],
                "Amount": float(txn["Amount"]),
                "Account": txn["Account"],
            },
            "outputs": {"category": txn["Category"]},
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


def _build_target(pool_archive):
    """Build the async target the experiment runs for each dataset example.

    Closes over `pool_archive` (the archive minus all held-out rows) so the
    agent's trailing-window/similarity lookup never sees a held-out row. Delegates
    to categorize_one, so windowing, the example prompt, structured output, and
    tracing are exactly the production path.
    """
    async def target(inputs: dict) -> dict:
        txn = {
            "Date": pd.Timestamp(inputs["Date"]),
            "Name": inputs["Name"],
            "Amount": inputs["Amount"],
            "Account": inputs["Account"],
        }
        try:
            r = await categorize_one(txn, pool_archive)
            return {"category": r.category, "reasoning": r.reasoning, "error": r.error}
        except Exception as e:  # noqa: BLE001
            # categorize_one *raises* on the recursion limit / missing
            # structured_response (agent.py). If that propagated out of the
            # target, LangSmith would mark the run errored and skip the
            # `correct` evaluator entirely — the example would show "No
            # feedback" and silently drop out of accuracy. Mirror agent_eval.py's
            # bounded() wrapper instead: return a None category so `correct`
            # scores it False and the example stays in the denominator.
            return {"category": None, "reasoning": None, "error": f"{type(e).__name__}: {e}"}

    return target


def correct(outputs: dict, reference_outputs: dict) -> bool:
    """Exact match: did the agent assign the archived ground-truth category?

    A None category (the agent errored) is scored False rather than skipped, so
    errors count against accuracy exactly as they do in agent_eval.py.
    """
    predicted = outputs.get("category")
    return predicted is not None and predicted == reference_outputs.get("category")


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
    args = parser.parse_args()

    archive = load_archive()

    # Same as agent_eval.py: rows with an unparseable Date can't be placed in a
    # month, so they are dropped rather than silently mixed into the pool.
    archive = archive[archive["Date"].notna()].reset_index(drop=True)

    held_out, n_candidates = select_held_out(
        archive, args.year, args.k, args.frac, args.seed
    )
    if held_out.empty:
        raise SystemExit(
            f"No archived transactions found for months 1..{args.k} of "
            f"{args.year}. Nothing to evaluate."
        )

    # The agent must never see a held-out row as an example; remove them all
    # globally so none leaks into another's example pool (matches agent_eval.py).
    pool_archive = archive.drop(held_out.index)

    client = Client()
    dataset_name = (
        f"txn-categorizer-{args.year}-k{args.k}-frac{args.frac}-seed{args.seed}"
    )
    dataset = sync_dataset(client, dataset_name, held_out)

    print(
        f"Dataset '{dataset_name}': {len(held_out)} held-out examples "
        f"(of {n_candidates} candidates in months 1..{args.k} of {args.year}); "
        f"example pool = {len(pool_archive)} rows. Running experiment..."
    )

    results = await client.aevaluate(
        _build_target(pool_archive),
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


if __name__ == "__main__":
    asyncio.run(main())
