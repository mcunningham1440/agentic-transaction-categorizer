"""web_search tool: merchant/brand lookup via Perplexity."""

import os

from perplexity import Perplexity

# Read at import — categorizer/__init__ has already loaded .env by this point.
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY", "")


def web_search_sync(query: str) -> str:
    if not PERPLEXITY_API_KEY:
        return "<error>PERPLEXITY_API_KEY env var not set</error>"

    client = Perplexity(api_key=PERPLEXITY_API_KEY)
    search = client.search.create(
        query=[query],
        max_results=5,
        max_tokens_per_page=1024,
        )

    if not search.results:
        return "<results></results>"

    blocks = []
    for i, r in enumerate(search.results, start=1):
        title = r.title or ""
        url = r.url or ""
        snippet = getattr(r, "snippet", None) or ""
        blocks.append(
            f'  <result index="{i}">\n'
            f"    <title>{title}</title>\n"
            f"    <url>{url}</url>\n"
            f"    <snippet>{snippet}</snippet>\n"
            f"  </result>"
        )

    inner = "\n".join(blocks)
    return f"<results>\n{inner}\n</results>"
