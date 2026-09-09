"""Fama-French factor returns from the Kenneth R. French data library.

Why this matters for the report
-------------------------------
A portfolio's excess return is not "alpha" until it has survived a factor
regression. Almost every naive long-only equity strategy that looks like it
generates alpha is in fact selling a known risk premium — small-cap, value,
profitability or momentum. The five-factor + momentum model is the standard
academic control set, so it is what this platform regresses against.

Source: Kenneth R. French Data Library, Tuck School of Business at Dartmouth.
Returns are published in percent; they are converted to decimals here once.
"""

from __future__ import annotations

import io
import logging
import re
import zipfile

import pandas as pd
import requests

from quantdesk.config import RAW_DIR

logger = logging.getLogger(__name__)

_FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FF5_ZIP = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOM_ZIP = "F-F_Momentum_Factor_daily_CSV.zip"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; QuantDesk research/1.0)"}

_FACTOR_DIR = RAW_DIR / "factors"
_CACHE_FILE = _FACTOR_DIR / "ff5_mom_daily.csv"

_DATE_ROW = re.compile(r"^\s*(\d{8})\s*,")


def _download_french_csv(archive: str, timeout: float = 60.0) -> str:
    """Return the raw CSV text inside one of the French library zip archives."""
    response = requests.get(_FRENCH_BASE + archive, headers=_HEADERS, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        member = next(n for n in zf.namelist() if n.lower().endswith((".csv", ".txt")))
        return zf.read(member).decode("utf-8", errors="replace")


def _parse_french_csv(text: str, columns: list[str]) -> pd.DataFrame:
    """Extract the daily block from a French library file.

    The files carry a free-text preamble and (for some vintages) an annual
    block appended after the daily one. Selecting on an 8-digit date token is
    the only parse that is stable across vintages.
    """
    records: list[list[str]] = []
    for line in text.splitlines():
        if _DATE_ROW.match(line):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= len(columns) + 1:
                records.append(parts[: len(columns) + 1])

    if not records:
        raise ValueError("No daily observations found in French library file")

    frame = pd.DataFrame(records, columns=["date", *columns])
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")
    frame = frame.set_index("date").astype(float)

    # -99.99 / -999 are the library's missing-value sentinels.
    frame = frame.mask(frame <= -99.0)
    return frame / 100.0  # percent -> decimal


def _download_factor_panel() -> pd.DataFrame:
    ff5 = _parse_french_csv(
        _download_french_csv(_FF5_ZIP), ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    )
    try:
        mom = _parse_french_csv(_download_french_csv(_MOM_ZIP), ["mom"])
        panel = ff5.join(mom, how="left")
    except Exception as exc:  # noqa: BLE001 - momentum is a nice-to-have
        logger.warning("Momentum factor unavailable (%s); continuing with FF5", exc)
        panel = ff5.assign(mom=pd.NA)

    panel.index.name = "date"
    return panel.dropna(subset=["mkt_rf"])


def load_factors(start: str, end: str, *, force: bool = False) -> pd.DataFrame:
    """Daily factor panel in decimals: mkt_rf, smb, hml, rmw, cma, mom, rf.

    Cached to disk so the repository reproduces offline.
    """
    _FACTOR_DIR.mkdir(parents=True, exist_ok=True)

    panel: pd.DataFrame | None = None
    if _CACHE_FILE.exists() and not force:
        panel = pd.read_csv(_CACHE_FILE, index_col="date", parse_dates=["date"])
        stale = len(pd.bdate_range(panel.index.max(), pd.Timestamp(end))) - 1
        # The library publishes with a lag of a few weeks; 45 business days of
        # tolerance avoids hammering the server on every run.
        if stale > 45:
            logger.info("Factor cache stale by %d business days — refreshing", stale)
            panel = None

    if panel is None:
        try:
            panel = _download_factor_panel()
            panel.to_csv(_CACHE_FILE, float_format="%.8f")
            logger.info("Factors downloaded: %d rows (%s → %s)", len(panel),
                        panel.index.min().date(), panel.index.max().date())
        except Exception as exc:  # noqa: BLE001
            if _CACHE_FILE.exists():
                logger.warning("Factor download failed (%s); using cache", exc)
                panel = pd.read_csv(_CACHE_FILE, index_col="date", parse_dates=["date"])
            else:
                raise RuntimeError(f"Factor data unavailable and no cache present: {exc}") from exc

    return panel.loc[str(start) : str(end)].sort_index()
