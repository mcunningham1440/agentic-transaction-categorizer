"""Entry point: held-out evaluation harness.

Thin wrapper so `python agent_eval.py` keeps working; the implementation lives
in categorizer.eval.
"""

import asyncio

from categorizer.eval import main

if __name__ == "__main__":
    asyncio.run(main())
