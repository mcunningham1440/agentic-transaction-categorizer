"""Entry point: end-to-end monthly categorization.

Thin wrapper so `python categorize_transactions.py` keeps working; the
implementation lives in categorizer.pipeline.
"""

import asyncio

from categorizer.pipeline import main

if __name__ == "__main__":
    asyncio.run(main())
