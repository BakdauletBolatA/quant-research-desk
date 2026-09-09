"""Walk-forward backtest engine.

What this engine does *not* do is as important as what it does.

* **No in-sample fitting.** At every rebalance the allocator sees a trailing
  window that ends on the rebalance date. Nothing is estimated on the full
  sample and then applied to it.
* **Weights drift between rebalances.** A quarterly-rebalanced portfolio is not
  a quarterly-rebalanced *equal-weight* portfolio; positions grow and shrink
  with prices, and the turnover figure only means something if that drift is
  modelled.
* **Costs are charged on traded notional**, on the day after the trade decision,
  at a configured bp rate. A backtest without costs is a marketing document:
  min-variance and momentum can look similar gross and diverge by hundreds of
  basis points a year net.
* **Rebalance dates are actual trading days**, taken from the price calendar,
  never from a synthetic month-end that may be a holiday.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantdesk.backtest.strategies import LABELS, StrategyContext, align_weights, get_strategy
from quantdesk.portfolio.optimizer import Constraints

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestResult:
    """Net-of-cost track record plus the full audit trail."""

    name: str
    label: str
    returns: pd.Series                 # net of transaction costs
    gross_returns: pd.Series
    weights: pd.DataFrame              # target weights at each rebalance
    turnover: pd.Series                # traded notional per rebalance
    costs: pd.Series                   # cost drag per day
    rebalance_dates: list[pd.Timestamp] = field(default_factory=list)

    @property
    def equity_curve(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    @property
    def annual_turnover(self) -> float:
        """Traded notional per year, as a multiple of portfolio value."""
        if self.turnover.empty:
            return np.nan
        years = (self.returns.index[-1] - self.returns.index[0]).days / 365.25
        return float(self.turnover.sum() / years) if years > 0 else np.nan

    @property
    def total_cost_drag(self) -> float:
        """Annualised return given up to transaction costs."""
        if self.returns.empty:
            return np.nan
        years = (self.returns.index[-1] - self.returns.index[0]).days / 365.25
        gross = float((1 + self.gross_returns).prod() ** (1 / years) - 1)
        net = float((1 + self.returns).prod() ** (1 / years) - 1)
        return gross - net

    @property
    def average_n_positions(self) -> float:
        if self.weights.empty:
            return np.nan
        return float((self.weights > 1e-6).sum(axis=1).mean())


def _rebalance_calendar(dates: pd.DatetimeIndex, rule: str) -> list[pd.Timestamp]:
    """Last actual trading day of each period in ``rule``."""
    anchors = pd.Series(dates, index=dates).resample(rule).last().dropna()
    return [pd.Timestamp(d) for d in anchors.to_numpy()]


def run_backtest(
    strategy_name: str,
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    risk_free: pd.Series,
    *,
    start: str,
    rebalance: str = "QE",
    lookback_days: int = 756,
    transaction_cost_bps: float = 10.0,
    constraints: Constraints | None = None,
    covariance_method: str = "ledoit_wolf",
    periods_per_year: int = 252,
    shares_outstanding: dict[str, float] | None = None,
    equity_risk_premium: float = 0.045,
    views: list[dict] | None = None,
    tau: float = 0.05,
    risk_aversion: float | None = None,
) -> BacktestResult:
    """Run one strategy walk-forward and return its net-of-cost track record."""
    strategy = get_strategy(strategy_name)
    constraints = constraints or Constraints()
    cost_rate = transaction_cost_bps / 10_000.0
    columns = returns.columns

    start_ts = pd.Timestamp(start)
    trading_days = returns.index
    live_days = trading_days[trading_days >= start_ts]
    if len(live_days) < periods_per_year:
        raise ValueError(f"Backtest window too short: {len(live_days)} days from {start}")

    rebalance_days = set(_rebalance_calendar(live_days, rebalance))
    # The first allocation happens on the first live day, not at the first
    # quarter end — otherwise the strategy sits in cash for up to a quarter.
    rebalance_days.add(live_days[0])

    weights: np.ndarray | None = None
    pending_cost: dict[pd.Timestamp, float] = {}
    gross_rows: list[tuple[pd.Timestamp, float]] = []
    net_rows: list[tuple[pd.Timestamp, float]] = []
    cost_rows: list[tuple[pd.Timestamp, float]] = []
    weight_history: dict[pd.Timestamp, pd.Series] = {}
    turnover_history: dict[pd.Timestamp, float] = {}

    for position, date in enumerate(live_days):
        # --- 1. earn today's return on yesterday's closing weights -----------
        if weights is not None:
            daily = returns.loc[date].to_numpy(dtype=float)
            gross = float(weights @ daily)
            cost = pending_cost.pop(date, 0.0)
            gross_rows.append((date, gross))
            net_rows.append((date, gross - cost))
            cost_rows.append((date, cost))
            # positions drift with prices until the next rebalance
            drifted = weights * (1.0 + daily)
            total = drifted.sum()
            weights = drifted / total if total > 0 else weights

        # --- 2. rebalance at today's close, using data up to today -----------
        if date in rebalance_days:
            window = returns.loc[:date].tail(lookback_days)
            if len(window) < max(126, lookback_days // 4):
                continue
            context = StrategyContext(
                window=window,
                prices=prices.loc[date],
                risk_free=risk_free.loc[:date].tail(lookback_days),
                constraints=constraints,
                covariance_method=covariance_method,
                periods_per_year=periods_per_year,
                shares_outstanding=shares_outstanding or {},
                equity_risk_premium=equity_risk_premium,
                views=views or [],
                tau=tau,
                risk_aversion=risk_aversion,
            )
            try:
                target = align_weights(strategy(context), columns)
            except Exception as exc:
                logger.warning("%s: allocation failed on %s (%s); holding previous weights",
                               strategy_name, date.date(), exc)
                continue

            previous = weights if weights is not None else np.zeros(len(columns))
            traded_notional = float(np.abs(target - previous).sum())
            turnover_history[date] = traded_notional
            weight_history[date] = pd.Series(target, index=columns)

            if position + 1 < len(live_days):
                pending_cost[live_days[position + 1]] = traded_notional * cost_rate
            weights = target

    net = pd.Series(dict(net_rows), name=strategy_name).sort_index()
    gross = pd.Series(dict(gross_rows), name=f"{strategy_name}_gross").sort_index()
    costs = pd.Series(dict(cost_rows), name="costs").sort_index()

    logger.info(
        "%-20s %d days, %d rebalances, annual turnover %.2fx, cost drag %.0f bp",
        strategy_name, len(net), len(weight_history),
        sum(turnover_history.values()) / max((net.index[-1] - net.index[0]).days / 365.25, 1e-9),
        (float((1 + gross).prod() / (1 + net).prod()) - 1) * 10_000
        / max((net.index[-1] - net.index[0]).days / 365.25, 1e-9),
    )

    return BacktestResult(
        name=strategy_name,
        label=LABELS.get(strategy_name, strategy_name),
        returns=net,
        gross_returns=gross,
        weights=pd.DataFrame(weight_history).T.sort_index(),
        turnover=pd.Series(turnover_history, name="turnover").sort_index(),
        costs=costs,
        rebalance_dates=sorted(weight_history),
    )


def run_all(
    strategy_names: list[str],
    returns: pd.DataFrame,
    prices: pd.DataFrame,
    risk_free: pd.Series,
    **kwargs,
) -> dict[str, BacktestResult]:
    """Run every configured strategy over the identical calendar and universe."""
    results: dict[str, BacktestResult] = {}
    for name in strategy_names:
        try:
            results[name] = run_backtest(name, returns, prices, risk_free, **kwargs)
        except Exception as exc:
            logger.error("Strategy %s failed: %s", name, exc)
    if not results:
        raise RuntimeError("Every strategy failed — check the configuration.")
    return results


def combine_returns(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Net return panel across strategies on the common calendar."""
    return pd.DataFrame({r.label: r.returns for r in results.values()}).dropna(how="all")


def turnover_table(results: dict[str, BacktestResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            r.label: {
                "Annual turnover (x)": r.annual_turnover,
                "Cost drag (ann.)": r.total_cost_drag,
                "Avg. positions": r.average_n_positions,
                "Rebalances": len(r.rebalance_dates),
            }
            for r in results.values()
        }
    ).T
