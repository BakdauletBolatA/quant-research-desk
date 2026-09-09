"""Allocation rules, expressed as pure functions of an estimation window.

Each strategy receives only data that existed at the rebalance date. That is
enforced by the engine, but the signatures are written so a look-ahead bug
would have to be deliberate: a strategy is handed a window of returns and the
prices as at the rebalance date, and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from quantdesk.portfolio.black_litterman import (
    black_litterman,
    derive_risk_aversion,
    market_cap_weights,
)
from quantdesk.portfolio.covariance import estimate_covariance
from quantdesk.portfolio.optimizer import Constraints, optimise


@dataclass(frozen=True)
class StrategyContext:
    """Everything a strategy is allowed to see at a rebalance date."""

    window: pd.DataFrame            # trailing returns, strictly up to the rebalance date
    prices: pd.Series               # last observed prices
    risk_free: pd.Series            # trailing risk-free path
    constraints: Constraints
    covariance_method: str
    periods_per_year: int
    shares_outstanding: dict[str, float]
    equity_risk_premium: float
    views: list[dict]
    tau: float
    risk_aversion: float | None = None

    def covariance(self) -> pd.DataFrame:
        return estimate_covariance(
            self.window, self.covariance_method, self.periods_per_year
        )

    def historical_excess_returns(self) -> pd.Series:
        """Annualised sample mean excess return — the classic bad estimator."""
        excess = self.window.sub(self.risk_free.reindex(self.window.index).ffill(), axis=0)
        return excess.mean() * self.periods_per_year


class Strategy(Protocol):
    def __call__(self, ctx: StrategyContext) -> pd.Series: ...


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def equal_weight(ctx: StrategyContext) -> pd.Series:
    return optimise("equal_weight", ctx.covariance(), constraints=ctx.constraints)


def min_variance(ctx: StrategyContext) -> pd.Series:
    return optimise("min_variance", ctx.covariance(), constraints=ctx.constraints)


def max_sharpe(ctx: StrategyContext) -> pd.Series:
    """Mean-variance on sample means. Included precisely because it is fragile."""
    return optimise(
        "max_sharpe", ctx.covariance(), ctx.historical_excess_returns(), ctx.constraints
    )


def risk_parity(ctx: StrategyContext) -> pd.Series:
    return optimise("risk_parity", ctx.covariance(), constraints=ctx.constraints)


def max_diversification(ctx: StrategyContext) -> pd.Series:
    return optimise("max_diversification", ctx.covariance(), constraints=ctx.constraints)


def hrp(ctx: StrategyContext) -> pd.Series:
    return optimise("hrp", ctx.covariance(), constraints=ctx.constraints)


def black_litterman_strategy(ctx: StrategyContext) -> pd.Series:
    """Equilibrium prior from cap weights at the rebalance date, plus fixed views.

    Simplification, stated rather than buried: share counts are taken from the
    current input sheet and applied to *historical* prices. Share counts move a
    few percent a year while prices move tens of percent, so the cap-weight
    ranking is dominated by the price path, which is point-in-time correct.
    """
    cov = ctx.covariance()
    weights = market_cap_weights(ctx.prices, ctx.shares_outstanding).reindex(cov.index)
    weights = weights / weights.sum()

    delta = ctx.risk_aversion or derive_risk_aversion(cov, weights, ctx.equity_risk_premium)
    posterior = black_litterman(cov, weights, ctx.views, risk_aversion=delta, tau=ctx.tau)
    return optimise(
        "max_sharpe", posterior.posterior_covariance, posterior.posterior_returns,
        ctx.constraints,
    ).rename("black_litterman")


def momentum_12_1(ctx: StrategyContext, n_hold: int = 5) -> pd.Series:
    """Cross-sectional momentum: top ``n_hold`` on 12-month return, skipping the
    most recent month to sidestep short-term reversal."""
    window = ctx.window
    if len(window) < 252:
        return equal_weight(ctx)

    formation = window.iloc[-252:-21]
    signal = (1.0 + formation).prod() - 1.0
    winners = signal.nlargest(min(n_hold, len(signal))).index

    weights = pd.Series(0.0, index=window.columns)
    weights[winners] = 1.0 / len(winners)

    cap = ctx.constraints.max_weight
    if weights.max() > cap:  # respect the concentration limit
        weights = weights.clip(upper=cap)
        weights = weights / weights.sum()
    return weights.rename("momentum_12_1")


REGISTRY: dict[str, Callable[[StrategyContext], pd.Series]] = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "max_sharpe": max_sharpe,
    "risk_parity": risk_parity,
    "max_diversification": max_diversification,
    "hrp": hrp,
    "black_litterman": black_litterman_strategy,
    "momentum_12_1": momentum_12_1,
}

LABELS = {
    "equal_weight": "Equal weight (1/N)",
    "min_variance": "Minimum variance",
    "max_sharpe": "Max Sharpe (sample μ)",
    "risk_parity": "Risk parity (ERC)",
    "max_diversification": "Max diversification",
    "hrp": "Hierarchical Risk Parity",
    "black_litterman": "Black-Litterman",
    "momentum_12_1": "Momentum 12-1",
}


def get_strategy(name: str) -> Callable[[StrategyContext], pd.Series]:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]


def align_weights(weights: pd.Series, columns: pd.Index) -> np.ndarray:
    """Reindex to the full universe and renormalise, defensively."""
    aligned = weights.reindex(columns).fillna(0.0).to_numpy(dtype=float)
    total = aligned.sum()
    return aligned / total if total > 0 else np.full(len(columns), 1.0 / len(columns))
