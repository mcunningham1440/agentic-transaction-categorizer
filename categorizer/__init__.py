"""Agentic transaction categorizer package.

Loading .env here (rather than in each entry point) guarantees environment
variables are populated before any submodule reads os.environ at import time
(e.g. tools.web's PERPLEXITY_API_KEY). Importing any categorizer.* module runs
this first, because Python executes a package's __init__ before its submodules.
"""

from dotenv import load_dotenv

from categorizer.paths import ENV_PATH

# If config/.env is absent (fresh machine), load_dotenv is a no-op and env vars
# must come from the real environment — same degradation as before the refactor.
load_dotenv(ENV_PATH)
