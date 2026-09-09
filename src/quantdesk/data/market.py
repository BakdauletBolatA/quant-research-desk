"""Daily price history with an offline-first disk cache.

Design notes
------------
* The public Yahoo Finance chart endpoint is used as the primary source. It is
  free and needs no key, but it is also not a contract — so every download is
  cached to ``data/raw/prices/<SYMBOL>.csv`` and the cache is committed. A
  reviewer who clones this repository can reproduce every number in the report
  with the network switched off.
* Adjusted close is the only series used downstream. Total-return adjustment
  (dividends + splits) is not optional: on a 10-year window an unadjusted
  price series understates a dividend payer such as XOM by several hundred
  basis points a year, which would silently corrupt every risk statistic.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from quantdesk.config import RAW_DIR

logger = logging.getLogger(__name__)

_CHART_HOSTS = ("query1.finance.yahoo.com", "query2.finance.yahoo.com")
_CHART_PATH = "/v8/finance/chart/{symbol}"
# A minimal, honest header set. The endpoint throttles chatty clients, so the
# session is reused across symbols and backs off on 429 rather than retrying hot.
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", "Accept": "*/*"}
_MAX_RETRIES = 4
_PRICE_DIR = RAW_DIR / "prices"
_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


@lru_cache(maxsize=1)
def _session() -> requests.Session:
    session = requests.Session()
    session.headers.clear()
    session.headers.update(_HEADERS)
    return session


def _get_json(symbol: str, params: dict[str, object], timeout: float) -> dict:
    """GET the chart endpoint with host failover and exponential backoff."""
    last_error: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        host = _CHART_HOSTS[attempt % len(_CHART_HOSTS)]
        url = f"https://{host}{_CHART_PATH.format(symbol=symbol)}"
        try:
            response = _session().get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(f"HTTP {response.status_code} from {host}")
                time.sleep(1.5 * 2**attempt)
                continue
            response.raise_for_status()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(1.5 * 2**attempt)
    raise RuntimeError(f"{symbol}: chart endpoint unavailable ({last_error})")


# ---------------------------------------------------------------------------
# Low-level download
# ---------------------------------------------------------------------------
def _to_epoch(date: str | dt.date) -> int:
    if isinstance(date, str):
        date = dt.date.fromisoformat(date)
    return int(dt.datetime.combine(date, dt.time(), tzinfo=dt.timezone.utc).timestamp())


def _fetch_one(symbol: str, start: str, end: str, timeout: float = 30.0) -> pd.DataFrame:
    """Fetch a single symbol's daily OHLCV + adjusted close."""
    params = {
        "period1": _to_epoch(start),
        "period2": _to_epoch(end) + 86_400,
        "interval": "1d",
        "events": "div,split",
        "includeAdjustedClose": "true",
    }
    payload = _get_json(symbol, params, timeout)

    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise RuntimeError(f"{symbol}: provider error {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"{symbol}: empty response from provider")

    result = results[0]
    timestamps = result.get("timestamp") or []
    if not timestamps:
        raise RuntimeError(f"{symbol}: no observations returned")

    quote = result["indicators"]["quote"][0]
    adj_block = result["indicators"].get("adjclose") or [{}]
    adj_close = adj_block[0].get("adjclose", quote["close"])

    frame = pd.DataFrame(
        {
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "adj_close": adj_close,
            "volume": quote.get("volume"),
        },
        index=pd.to_datetime(pd.Series(timestamps), unit="s", utc=True),
    )
    frame.index = frame.index.tz_convert("America/New_York").normalize().tz_localize(None)
    frame.index.name = "date"
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.dropna(subset=["adj_close"])


# ---------------------------------------------------------------------------
# Cache-aware public API
# ---------------------------------------------------------------------------
def _cache_path(symbol: str) -> Path:
    return _PRICE_DIR / f"{symbol.replace('/', '-')}.csv"


def _read_cache(symbol: str) -> pd.DataFrame | None:
    path = _cache_path(symbol)
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return frame if not frame.empty else None


def _cache_is_fresh(frame: pd.DataFrame, end: str, max_stale_days: int) -> bool:
    target = pd.Timestamp(end)
    # Only trading days count; a Friday cache is still fresh on a Sunday.
    gap = len(pd.bdate_range(frame.index.max(), target)) - 1
    return gap <= max_stale_days


def download_prices(
    symbols: list[str],
    start: str,
    end: str,
    *,
    max_stale_days: int = 5,
    force: bool = False,
    pause: float = 0.35,
) -> dict[str, pd.DataFrame]:
    """Return per-symbol OHLCV frames, downloading only what the cache lacks.

    Network failures are never fatal when a cached copy exists — the pipeline
    degrades to the cached history and says so in the log.
    """
    _PRICE_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        cached = None if force else _read_cache(symbol)
        if cached is not None and _cache_is_fresh(cached, end, max_stale_days):
            logger.info("%-6s cache hit  (%s → %s)", symbol, cached.index.min().date(),
                        cached.index.max().date())
            out[symbol] = cached
            continue

        try:
            frame = _fetch_one(symbol, start, end)
            frame.to_csv(_cache_path(symbol), float_format="%.6f")
            logger.info("%-6s downloaded (%d rows, %s → %s)", symbol, len(frame),
                        frame.index.min().date(), frame.index.max().date())
            out[symbol] = frame
            time.sleep(pause)  # be a polite client
        except Exception as exc:  # noqa: BLE001 - degrade, never crash the pipeline
            if cached is not None:
                logger.warning("%-6s download failed (%s); using cached history", symbol, exc)
                out[symbol] = cached
            else:
                logger.error("%-6s download failed and no cache available: %s", symbol, exc)

    if not out:
        raise RuntimeError("No price data available: network unreachable and cache empty.")
    return out


def load_price_panel(
    symbols: list[str],
    start: str,
    end: str,
    *,
    field: str = "adj_close",
    max_stale_days: int = 5,
    force: bool = False,
) -> pd.DataFrame:
    """Wide price panel (rows = dates, columns = symbols) on the common calendar."""
    if field not in _COLUMNS:
        raise ValueError(f"field must be one of {_COLUMNS}, got {field!r}")

    frames = download_prices(symbols, start, end, max_stale_days=max_stale_days, force=force)
    panel = pd.DataFrame({sym: df[field] for sym, df in frames.items()})
    panel = panel.loc[str(start) : str(end)]

    # Align to the benchmark-quality calendar: keep dates where at least 80% of
    # the universe traded, then forward-fill isolated holiday gaps (max 3 days).
    coverage = panel.notna().mean(axis=1)
    panel = panel.loc[coverage >= 0.8]
    panel = panel.ffill(limit=3).dropna(how="any")

    panel.index.name = "date"
    return panel[[s for s in symbols if s in panel.columns]].sort_index()
