"""The research panel: one aligned object every downstream module consumes.

Aligning prices, factors and the risk-free rate *once*, in one place, is the
single most effective defence against the classic silent bug in performance
work — a Sharpe ratio computed against a risk-free series that is off by a day,
or a factor regression run on a mis-joined calendar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantdesk.config import Config, load_config
from quantdesk.data.factors import load_factors
from quantdesk.data.market import load_price_panel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResearchPanel:
    """Immutable, calendar-aligned research dataset."""

    prices: pd.DataFrame          # universe adjusted closes
    returns: pd.DataFrame         # universe simple daily returns
    benchmark_prices: pd.Series
    benchmark_returns: pd.Series
    risk_free: pd.Series          # daily simple risk-free rate
    factors: pd.DataFrame         # mkt_rf, smb, hml, rmw, cma, mom
    sectors: dict[str, str]
    names: dict[str, str]
    periods_per_year: int = 252

    # -- derived views --------------------------------------------------------
    @property
    def excess_returns(self) -> pd.DataFrame:
        return self.returns.sub(self.risk_free, axis=0)

    @property
    def benchmark_excess_returns(self) -> pd.Series:
        return self.benchmark_returns.sub(self.risk_free)

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def start(self) -> pd.Timestamp:
        return self.returns.index.min()

    @property
    def end(self) -> pd.Timestamp:
        return self.returns.index.max()

    @property
    def n_years(self) -> float:
        return len(self.returns) / self.periods_per_year

    def summary(self) -> pd.DataFrame:
        """Per-name coverage and liquidity sanity table."""
        ann = self.periods_per_year
        rows = {
            "sector": pd.Series(self.sectors),
            "obs": self.returns.notna().sum(),
            "first": self.prices.apply(lambda s: s.first_valid_index()),
            "last": self.prices.apply(lambda s: s.last_valid_index()),
            "ann_return": (1 + self.returns).prod() ** (ann / len(self.returns)) - 1,
            "ann_vol": self.returns.std(ddof=1) * np.sqrt(ann),
        }
        return pd.DataFrame(rows).loc[self.tickers]


def build_panel(cfg: Config | None = None, *, force: bool = False) -> ResearchPanel:
    """Download (or load from cache) and align every input series."""
    cfg = cfg or load_config()
    data_cfg = cfg["data"]
    start, end = cfg.start, cfg.end
    benchmark = cfg.benchmark

    prices = load_price_panel(
        cfg.all_symbols,
        start,
        end,
        max_stale_days=int(data_cfg.get("max_stale_days", 5)),
        force=force,
    )
    missing = [s for s in cfg.all_symbols if s not in prices.columns]
    if missing:
        logger.warning("Dropped from universe (no data): %s", ", ".join(missing))
    if benchmark not in prices.columns:
        raise RuntimeError(f"Benchmark {benchmark} unavailable — cannot proceed.")

    factors_raw = load_factors(start, end, force=force)

    # The market calendar is authoritative. The French library publishes with a
    # multi-week lag, so factors are *reindexed onto* the price calendar rather
    # than truncating it — otherwise the most recent quarter of performance
    # would silently disappear from the report.
    overlap = prices.index.intersection(factors_raw.index)
    if len(overlap) < 250:
        raise RuntimeError(
            f"Only {len(overlap)} overlapping days between prices and factors — "
            "the factor library is badly stale or the price calendar is wrong."
        )
    lag_days = int((prices.index.max() - factors_raw.index.max()).days)
    if lag_days > 0:
        logger.info(
            "Factor library lags prices by %d calendar days (last factor obs %s); "
            "factor-based statistics use the overlapping window only.",
            lag_days, factors_raw.index.max().date(),
        )

    factors = factors_raw.reindex(prices.index)
    returns = prices.pct_change().iloc[1:]
    factors = factors.iloc[1:]

    # Risk-free: carry the last published daily rate forward over the lag. The
    # alternative — dropping those days — would bias the Sharpe ratio upwards.
    risk_free = factors["rf"].astype(float).ffill().bfill()

    universe = [t for t in cfg.tickers if t in returns.columns]
    panel = ResearchPanel(
        prices=prices[universe],
        returns=returns[universe],
        benchmark_prices=prices[benchmark],
        benchmark_returns=returns[benchmark],
        risk_free=risk_free,
        factors=factors[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]],
        sectors={t: cfg.sectors[t] for t in universe},
        names={t: cfg.names[t] for t in universe},
        periods_per_year=cfg.periods_per_year,
    )
    logger.info(
        "Panel ready: %d names, %d days (%s → %s)",
        len(universe), len(panel.returns), panel.start.date(), panel.end.date(),
    )
    return panel
