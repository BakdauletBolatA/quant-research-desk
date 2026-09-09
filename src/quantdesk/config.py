"""Configuration loading and project paths.

The whole platform is configuration-driven: research assumptions live in YAML,
never in code, so that a reviewer can diff an assumption change the same way
they diff a bug fix.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to."""
    for path in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR, FIGURES_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Immutable view over ``config/config.yaml``."""

    raw: dict[str, Any] = field(repr=False)

    # -- convenience accessors ------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def tickers(self) -> list[str]:
        return list(self.raw["universe"].keys())

    @property
    def benchmark(self) -> str:
        return self.raw["data"]["benchmark"]

    @property
    def all_symbols(self) -> list[str]:
        """Universe plus the benchmark, de-duplicated, order preserved."""
        symbols = [*self.tickers, self.benchmark]
        seen: set[str] = set()
        return [s for s in symbols if not (s in seen or seen.add(s))]

    @property
    def sectors(self) -> dict[str, str]:
        return {t: meta["sector"] for t, meta in self.raw["universe"].items()}

    @property
    def names(self) -> dict[str, str]:
        return {t: meta["name"] for t, meta in self.raw["universe"].items()}

    @property
    def start(self) -> str:
        return self.raw["data"]["start"]

    @property
    def end(self) -> str:
        end = self.raw["data"].get("end")
        return end or dt.date.today().isoformat()

    @property
    def periods_per_year(self) -> int:
        return int(self.raw["analytics"]["periods_per_year"])


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> Config:
    """Load and cache the master configuration."""
    cfg_path = Path(path) if path else CONFIG_DIR / "config.yaml"
    with open(cfg_path, encoding="utf-8") as fh:
        return Config(raw=yaml.safe_load(fh))


@lru_cache(maxsize=4)
def load_fundamentals(path: str | Path | None = None) -> dict[str, Any]:
    """Load the analyst input sheet used by the valuation models."""
    fund_path = Path(path) if path else CONFIG_DIR / "fundamentals.yaml"
    with open(fund_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
