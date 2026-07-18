"""Browser-driven transaction downloaders."""

from categorizer.downloads.chase import ChaseSource

SOURCES = {ChaseSource.slug: ChaseSource}

__all__ = ["SOURCES", "ChaseSource"]
