"""Command-line interface for browser-driven bank downloads."""

from __future__ import annotations

import argparse
import asyncio
from datetime import date

from categorizer.downloads import SOURCES
from categorizer.downloads.browser import BrowserDownloader
from categorizer.downloads.source import DownloadError
from categorizer.paths import PROJECT_ROOT


def _previous_month(today: date | None = None) -> tuple[int, int]:
    today = today or date.today()
    if today.month == 1:
        return 12, today.year - 1
    return today.month - 1, today.year


def _parser() -> argparse.ArgumentParser:
    default_month, default_year = _previous_month()
    parser = argparse.ArgumentParser(
        description="Download monthly bank transaction CSVs with a supervised browser agent."
    )
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--month", type=int, default=default_month)
    parser.add_argument("--year", type=int, default=default_year)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="after validating the new CSV, remove older CSVs for this source/month folder",
    )
    parser.add_argument("--max-turns", type=int, default=60)
    return parser


async def _run(args: argparse.Namespace) -> None:
    if not 1 <= args.month <= 12:
        raise DownloadError("--month must be between 1 and 12")
    year = args.year + 2000 if 0 <= args.year <= 99 else args.year
    if not 2000 <= year <= 2100:
        raise DownloadError("--year must be a two- or four-digit year from 2000 to 2100")

    source = SOURCES[args.source]()
    settings = source.settings_from_env()
    downloader = BrowserDownloader(
        source=source,
        settings=settings,
        project_root=PROJECT_ROOT,
        max_turns=args.max_turns,
    )
    path = await downloader.download(args.month, year, replace=args.replace)
    print(f"\nValidated {source.display_name} CSV: {path}")


def main() -> None:
    args = _parser().parse_args()
    try:
        asyncio.run(_run(args))
    except (DownloadError, KeyboardInterrupt) as exc:
        raise SystemExit(f"Download failed: {exc}") from exc


if __name__ == "__main__":
    main()
