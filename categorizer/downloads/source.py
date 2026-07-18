"""Contracts shared by bank-specific transaction downloaders."""

from __future__ import annotations

import csv
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class DownloadError(RuntimeError):
    """Raised when a source cannot safely produce a usable CSV."""


class KnownPageState(str, Enum):
    """Result of a source-specific deterministic browser shortcut."""

    NO_MATCH = "no_match"
    ADVANCED = "advanced"
    HUMAN_REQUIRED = "human_required"


@dataclass(frozen=True)
class SourceSettings:
    start_url: str
    username: str
    password: str


class TransactionSource(ABC):
    """Bank-specific policy consumed by the generic browser runner."""

    slug: str
    display_name: str
    filename_prefix: str
    profile_dirname: str
    allowed_domains: tuple[str, ...]
    required_columns: frozenset[str]

    @classmethod
    def settings_from_env(cls) -> SourceSettings:
        prefix = cls.slug.upper().replace("-", "_")
        values = {
            "URL": os.environ.get(f"{prefix}_URL", "").strip(),
            "USERNAME": os.environ.get(f"{prefix}_USERNAME", "").strip(),
            "PASSWORD": os.environ.get(f"{prefix}_PASSWORD", ""),
        }
        missing = [f"{prefix}_{name}" for name, value in values.items() if not value]
        if missing:
            raise DownloadError(
                f"Missing required setting(s): {', '.join(missing)}. "
                "Add them to config/.env (see config/.env.sample)."
            )

        source = cls()
        source.validate_url(values["URL"])
        return SourceSettings(
            start_url=values["URL"],
            username=values["USERNAME"],
            password=values["PASSWORD"],
        )

    def validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "https" or not self.is_allowed_host(host):
            allowed = ", ".join(self.allowed_domains)
            raise DownloadError(
                f"Refusing {self.display_name} URL outside its HTTPS allowlist "
                f"({allowed}): {url}"
            )

    def is_allowed_host(self, host: str) -> bool:
        host = host.lower().rstrip(".")
        return any(
            host == domain or host.endswith(f".{domain}")
            for domain in self.allowed_domains
        )

    def normalized_filename(self, suggested_filename: str) -> str:
        name = Path(suggested_filename).name
        if not name.lower().endswith(".csv"):
            name = f"{name}.csv"
        if not name.lower().startswith(self.filename_prefix.lower()):
            name = f"{self.filename_prefix}_{name}"
        return name

    def validate_csv(self, path: Path) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise DownloadError(f"Downloaded file is empty or missing: {path}")
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                header = next(csv.reader(handle))
        except (OSError, StopIteration, UnicodeDecodeError, csv.Error) as exc:
            raise DownloadError(f"Downloaded file is not a readable CSV: {path}") from exc

        missing = self.required_columns.difference(header)
        if missing:
            raise DownloadError(
                f"Downloaded CSV is missing expected {self.display_name} column(s): "
                f"{', '.join(sorted(missing))}"
            )

    async def handle_known_page(
        self, page: Any, settings: SourceSettings
    ) -> KnownPageState:
        """Handle a stable source-specific page without using a model turn."""

        return KnownPageState.NO_MATCH

    @abstractmethod
    def task(self, settings: SourceSettings, month: int, year: int) -> str:
        """Return the source-specific computer-use task."""
