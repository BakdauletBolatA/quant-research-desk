"""Portfolio construction.

Six allocators, chosen so that the report can separate *what the optimiser
does* from *what the estimates do*:

``equal_weight``        No estimation at all. The benchmark every clever method
                        has to beat, and frequently does not (DeMiguel, Garlappi
                        & Uppal, 2009).
``min_variance``        Uses only the covariance matrix. No expected returns,
                        which is the input nobody can forecast.
``max_sharpe``          Uses expected returns too, and is therefore the most
                        estimation-error-sensitive method here. Included partly
                        so the backtest can show that.
``risk_parity``         Equal risk contribution via the convex log-barrier
                        formulation, which has a unique solution and does not
                        depend on a starting guess.
``max_diversification`` Maximises the ratio of weighted average volatility to
                        portfolio volatility.
``hrp``                 Hierarchical Risk Parity (Lopez de Prado, 2016).
                        Never inverts the covariance matrix, so it is immune to
                        the ill-conditioning that wrecks mean-variance on short
                        estimation windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from quantdesk.portfolio.covariance import correlation_from_covariance

METHODS = (
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "risk_parity",
    "max_diversification",
    "hrp",
)

_SLSQP_OPTIONS = {"maxiter": 500, "ftol": 1e-12}


@dataclass(frozen=True)
class Constraints:
    """Long-only box constraints plus full investment."""

    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0

    def bounds(self, n: int) -> list[tuple[float, float]]:
        lower = max(self.min_weight, 0.0) if self.long_only else self.min_weight
        upper = self.max_weight
        if upper * n < 1.0 - 1e-9:
            raise ValueError(
                f"max_weight={upper} cannot be satisfied with {n} assets "
                f"(needs at least {1.0 / n:.4f})"
            )
        return [(lower, upper)] * n


# ---------------------------------------------------------------------------
# Portfolio maths
# ---------------------------------------------------------------------------
def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(max(weights @ cov @ weights, 0.0)))


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Euler decomposition: each asset's share of total portfolio risk."""
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return np.zeros_like(weights)
    return weights * (cov @ weights) / vol


def diversification_ratio(weights: np.ndarray, cov: np.ndarray) -> float:
    vol = portfolio_volatility(weights, cov)
    if vol == 0:
        return np.nan
    return float((weights @ np.sqrt(np.diag(cov))) / vol)


def effective_number_of_bets(weights: np.ndarray) -> float:
    """Inverse Herfindahl — 16 equal positions score 16, a single position scores 1."""
    total = np.sum(weights**2)
    return float(1.0 / total) if total > 0 else np.nan


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------
def _normalise(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    return weights / total if total != 0 else weights


def _solve(objective, n: int, constraints: Constraints, x0: np.ndarray | None = None):
    bounds = constraints.bounds(n)
    budget = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    start = x0 if x0 is not None else np.full(n, 1.0 / n)
    result = minimize(
        objective, start, method="SLSQP", bounds=bounds, constraints=[budget],
        options=_SLSQP_OPTIONS,
    )
    if not result.success:
        # SLSQP occasionally stalls on a flat ftol; a perturbed restart is
        # cheaper and more honest than silently returning a failed iterate.
        rng = np.random.default_rng(0)
        perturbed = _normalise(np.abs(rng.normal(1.0, 0.25, n)))
        result = minimize(
            objective, perturbed, method="SLSQP", bounds=bounds, constraints=[budget],
            options=_SLSQP_OPTIONS,
        )
    if not result.success:
        raise RuntimeError(f"Optimiser failed: {result.message}")
    return _normalise(np.clip(result.x, bounds[0][0], bounds[0][1]))


def equal_weight(cov: pd.DataFrame, **_) -> pd.Series:
    n = cov.shape[0]
    return pd.Series(np.full(n, 1.0 / n), index=cov.index, name="equal_weight")


def min_variance(cov: pd.DataFrame, constraints: Constraints | None = None, **_) -> pd.Series:
    constraints = constraints or Constraints()
    matrix = cov.to_numpy()
    weights = _solve(lambda w: w @ matrix @ w, cov.shape[0], constraints)
    return pd.Series(weights, index=cov.index, name="min_variance")


def max_sharpe(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    constraints: Constraints | None = None,
    **_,
) -> pd.Series:
    """Maximise the ex-ante Sharpe ratio of annualised *excess* returns."""
    constraints = constraints or Constraints()
    matrix = cov.to_numpy()
    mu = expected_returns.reindex(cov.index).to_numpy(dtype=float)

    def negative_sharpe(w: np.ndarray) -> float:
        vol = portfolio_volatility(w, matrix)
        return 1e6 if vol <= 0 else float(-(w @ mu) / vol)

    weights = _solve(negative_sharpe, cov.shape[0], constraints)
    return pd.Series(weights, index=cov.index, name="max_sharpe")


def max_diversification(
    cov: pd.DataFrame, constraints: Constraints | None = None, **_
) -> pd.Series:
    constraints = constraints or Constraints()
    matrix = cov.to_numpy()
    sigma = np.sqrt(np.diag(matrix))

    def negative_dr(w: np.ndarray) -> float:
        vol = portfolio_volatility(w, matrix)
        return 1e6 if vol <= 0 else float(-(w @ sigma) / vol)

    weights = _solve(negative_dr, cov.shape[0], constraints)
    return pd.Series(weights, index=cov.index, name="max_diversification")


def risk_parity(
    cov: pd.DataFrame,
    constraints: Constraints | None = None,
    budgets: np.ndarray | None = None,
    **_,
) -> pd.Series:
    """Equal risk contribution via the convex log-barrier formulation.

        min  0.5 w'Σw − Σ b_i ln(w_i)      subject to w > 0

    The solution is unique and scale-free; normalising afterwards gives the
    fully invested ERC portfolio. This is far more reliable than minimising the
    squared dispersion of risk contributions, which is non-convex and start-
    point dependent.
    """
    matrix = cov.to_numpy()
    n = matrix.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)

    def objective(w: np.ndarray) -> float:
        return 0.5 * float(w @ matrix @ w) - float(b @ np.log(w))

    def gradient(w: np.ndarray) -> np.ndarray:
        return matrix @ w - b / w

    result = minimize(
        objective, np.full(n, 1.0 / n), jac=gradient, method="L-BFGS-B",
        bounds=[(1e-9, np.inf)] * n, options={"maxiter": 1000, "ftol": 1e-16},
    )
    weights = _normalise(result.x)

    # Box constraints are applied after the fact: the log-barrier form has no
    # room for them, and clipping an ERC solution is a documented compromise
    # rather than a different portfolio.
    if constraints is not None and constraints.max_weight < 1.0:
        weights = _normalise(np.clip(weights, constraints.min_weight, constraints.max_weight))

    return pd.Series(weights, index=cov.index, name="risk_parity")


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------
def _inverse_variance_weights(cov: pd.DataFrame) -> np.ndarray:
    ivp = 1.0 / np.diag(cov.to_numpy())
    return ivp / ivp.sum()


def _cluster_variance(cov: pd.DataFrame, items: list[str]) -> float:
    sub = cov.loc[items, items]
    w = _inverse_variance_weights(sub)
    return float(w @ sub.to_numpy() @ w)


def _quasi_diagonal_order(link: np.ndarray, n_items: int) -> list[int]:
    """Reorder leaves so that similar assets sit next to each other."""
    link = link.astype(int)
    order = pd.Series([link[-1, 0], link[-1, 1]], dtype=int)

    while order.max() >= n_items:
        order.index = range(0, order.shape[0] * 2, 2)
        clusters = order[order >= n_items]
        positions = clusters.index
        cluster_ids = clusters.to_numpy() - n_items
        order[positions] = link[cluster_ids, 0]
        right = pd.Series(link[cluster_ids, 1], index=positions + 1)
        order = pd.concat([order, right]).sort_index()
        order.index = range(order.shape[0])

    return order.astype(int).tolist()


def hrp(cov: pd.DataFrame, constraints: Constraints | None = None, **_) -> pd.Series:
    """Hierarchical Risk Parity: cluster, quasi-diagonalise, bisect."""
    corr = correlation_from_covariance(cov)
    # Lopez de Prado's correlation distance: a proper metric on correlations.
    distance = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, 1.0))
    np.fill_diagonal(distance, 0.0)
    link = linkage(squareform(distance, checks=False), method="single")

    order = _quasi_diagonal_order(link, cov.shape[0])
    ordered = [cov.index[i] for i in order]

    weights = pd.Series(1.0, index=ordered)
    clusters: list[list[str]] = [ordered]
    while clusters:
        clusters = [
            group[start:stop]
            for group in clusters
            for start, stop in ((0, len(group) // 2), (len(group) // 2, len(group)))
            if len(group) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_left, var_right = _cluster_variance(cov, left), _cluster_variance(cov, right)
            total = var_left + var_right
            alpha = 1.0 - var_left / total if total > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha

    weights = weights.reindex(cov.index)
    if constraints is not None and constraints.max_weight < 1.0:
        weights = pd.Series(
            _normalise(np.clip(weights.to_numpy(), constraints.min_weight,
                               constraints.max_weight)),
            index=cov.index,
        )
    weights.name = "hrp"
    return weights


# ---------------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------------
def efficient_frontier(
    cov: pd.DataFrame,
    expected_returns: pd.Series,
    constraints: Constraints | None = None,
    n_points: int = 40,
) -> pd.DataFrame:
    """Minimum-variance portfolio for a grid of target returns."""
    constraints = constraints or Constraints()
    matrix = cov.to_numpy()
    mu = expected_returns.reindex(cov.index).to_numpy(dtype=float)
    bounds = constraints.bounds(cov.shape[0])

    lo = float(min_variance(cov, constraints) @ mu)
    hi = float(mu.max() * min(constraints.max_weight * len(mu), 1.0))
    targets = np.linspace(lo, max(hi, lo + 1e-6), n_points)

    rows = []
    for target in targets:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        result = minimize(
            lambda w: w @ matrix @ w, np.full(len(mu), 1.0 / len(mu)), method="SLSQP",
            bounds=bounds, constraints=cons, options=_SLSQP_OPTIONS,
        )
        if not result.success:
            continue
        w = _normalise(np.clip(result.x, bounds[0][0], bounds[0][1]))
        vol = portfolio_volatility(w, matrix)
        rows.append(
            {"volatility": vol, "expected_return": float(w @ mu),
             "sharpe": float(w @ mu) / vol if vol > 0 else np.nan}
        )

    return pd.DataFrame(rows).drop_duplicates(subset="volatility").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
_DISPATCH = {
    "equal_weight": equal_weight,
    "min_variance": min_variance,
    "max_sharpe": max_sharpe,
    "risk_parity": risk_parity,
    "max_diversification": max_diversification,
    "hrp": hrp,
}


def optimise(
    method: str,
    cov: pd.DataFrame,
    expected_returns: pd.Series | None = None,
    constraints: Constraints | None = None,
) -> pd.Series:
    """Build a portfolio with the named allocator."""
    if method not in _DISPATCH:
        raise ValueError(f"method must be one of {tuple(_DISPATCH)}, got {method!r}")
    if method == "max_sharpe" and expected_returns is None:
        raise ValueError("max_sharpe requires expected_returns")
    return _DISPATCH[method](
        cov=cov, expected_returns=expected_returns, constraints=constraints
    ).rename(method)
