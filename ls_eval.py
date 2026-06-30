"""Entry point: LangSmith Datasets + Experiments eval harness.

Thin wrapper so `python ls_eval.py` keeps working; the implementation lives in
categorizer.ls_eval. Companion to agent_eval.py — same held-out selection, but
results land in LangSmith's Datasets & Experiments tab instead of a local CSV.
"""

import asyncio

from categorizer.ls_eval import main

if __name__ == "__main__":
    asyncio.run(main())
