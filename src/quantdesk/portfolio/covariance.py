"""Covariance estimation.

The sample covariance matrix is the default in textbooks and a liability in
practice. With N = 16 names and T = 3,700 days it is well conditioned, but the
optimiser still keys on the smallest eigenvalues — precisely the directions
estimated with the least precision — and answers with a portfolio that is long
and short the estimation error. With the 3-year (T = 756) windows used in the
walk-forward backtest the problem is materially worse.

Three estimators are provided:

* ``sample``       — the strawman, kept so the report can quantify the damage.
* ``ledoit_wolf``  — optimal linear shrinkage toward a scaled identity. Closed
                     form, no tuning parameter, and it dominates the sample
                     matrix out of sample by a wide margin.
* ``ewma``         — RiskMetrics exponential weighting, for a covariance that
                     tracks the current regime rather than the last decade.

Every estimator is passed through a PSD repair so that downstream optimisers
never receive a matrix with a (numerically) negative eigenvalue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

COVARIANCE_METHODS = ("sample", "ledoit_wolf", "ewma")


def nearest_psd(matrix: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Clip negative eigenvalues to ``epsilon`` and rebuild a symmetric matrix."""
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if (eigenvalues > 0).all():
        return symmetric
    clipped = np.clip(eigenvalues, epsilon, None)
    repaired = eigenvectors @ np.diag(clipped) @ eigenvectors.T
    return (repaired + repaired.T) / 2.0


def sample_covariance(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    cov = returns.cov(ddof=1).to_numpy() * periods
    return pd.DataFrame(nearest_psd(cov), index=returns.columns, columns=returns.columns)


def ledoit_wolf_covariance(returns: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage toward a scaled identity, annualised."""
    clean = returns.dropna(how="any")
    estimator = LedoitWolf(assume_centered=False).fit(clean.to_numpy())
    cov = nearest_psd(estimator.covariance_ * periods)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> float:
    """The shrinkage intensity chosen by the estimator (0 = sample, 1 = identity)."""
    clean = returns.dropna(how="any")
    return float(LedoitWolf(assume_centered=False).fit(clean.to_numpy()).shrinkage_)


def ewma_covariance(returns: pd.DataFrame, lam: float = 0.94, periods: int = 252) -> pd.DataFrame:
    """Exponentially weighted covariance using the RiskMetrics decay."""
    if not 0 < lam < 1:
        raise ValueError("lam must lie strictly between 0 and 1")
    clean = returns.dropna(how="any")
    values = clean.to_numpy()
    n_obs, n_assets = values.shape
    if n_obs < 2:
        raise ValueError("Need at least two observations")

    demeaned = values - values.mean(axis=0)
    weights = (1.0 - lam) * lam ** np.arange(n_obs - 1, -1, -1)
    weights /= weights.sum()

    cov = np.zeros((n_assets, n_assets))
    for w, row in zip(weights, demeaned, strict=True):
        cov += w * np.outer(row, row)

    cov = nearest_psd(cov * periods)
    return pd.DataFrame(cov, index=clean.columns, columns=clean.columns)


def estimate_covariance(
    returns: pd.DataFrame, method: str = "ledoit_wolf", periods: int = 252, lam: float = 0.94
) -> pd.DataFrame:
    """Dispatch to the configured estimator."""
    if method not in COVARIANCE_METHODS:
        raise ValueError(f"method must be one of {COVARIANCE_METHODS}, got {method!r}")
    if method == "sample":
        return sample_covariance(returns, periods)
    if method == "ewma":
        return ewma_covariance(returns, lam, periods)
    return ledoit_wolf_covariance(returns, periods)


def condition_number(cov: pd.DataFrame) -> float:
    """Ratio of largest to smallest eigenvalue — how invertible the matrix really is."""
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    smallest = float(eigenvalues.min())
    return float(eigenvalues.max() / smallest) if smallest > 0 else np.inf


def correlation_from_covariance(cov: pd.DataFrame) -> pd.DataFrame:
    sigma = np.sqrt(np.diag(cov.to_numpy()))
    corr = cov.to_numpy() / np.outer(sigma, sigma)
    np.fill_diagonal(corr, 1.0)
    return pd.DataFrame(np.clip(corr, -1.0, 1.0), index=cov.index, columns=cov.columns)
