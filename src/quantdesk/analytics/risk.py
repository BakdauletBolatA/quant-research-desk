"""Market risk: VaR, Expected Shortfall, and regulator-style model validation.

Sign convention: every VaR / ES figure is returned as a **positive loss
magnitude** at the stated confidence. A 1-day 99% VaR of 0.031 means "on the
worst 1 day in 100 we expect to lose at least 3.1%".

Four estimators are implemented because they disagree, and the disagreement is
the interesting part:

* ``historical``      — empirical quantile. No distributional assumption, but
                        it can only see losses that have already happened.
* ``gaussian``        — closed form. Systematically understates equity tail
                        risk; kept as the strawman everyone benchmarks against.
* ``cornish_fisher``  — Gaussian quantile expanded for skew and excess
                        kurtosis. Usually the best cheap estimator.
* ``ewma``            — RiskMetrics conditional volatility (λ = 0.94). Reacts
                        to regime changes in days rather than in a year, but
                        still assumes a normal conditional distribution.
* ``filtered_historical`` — Barone-Adesi FHS: standardise returns by their
                        conditional volatility, take the *empirical* quantile
                        of the standardised residuals, then rescale by today's
                        volatility. Conditional like EWMA, fat-tailed like the
                        historical method. This is the one that passes.

The point of ``var_backtest`` is that a VaR number nobody has backtested is a
decoration. Kupiec's proportion-of-failures test and Christoffersen's
independence test are the two the Basel traffic-light framework is built on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

VAR_METHODS = ("historical", "gaussian", "cornish_fisher", "ewma", "filtered_historical")

# The Cornish-Fisher expansion is only monotonic — and therefore only a valid
# quantile mapping — inside a bounded skew/kurtosis domain. A full-sample equity
# series that contains February 2020 has excess kurtosis around 15, well outside
# it, and the raw expansion then returns a nonsensically large VaR. Clamping the
# moments to the validity domain is the standard practitioner fix; the
# alternative (using the raw moments) produces a number no risk committee would
# sign. See Maillard (2018) on the CF domain of validity.
_CF_MAX_SKEW = 1.5
_CF_MAX_EXCESS_KURT = 6.0


def _as_series(returns: pd.Series | np.ndarray) -> pd.Series:
    series = returns if isinstance(returns, pd.Series) else pd.Series(returns)
    return series.astype(float).dropna()


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def _ewma_variance(returns: pd.Series, lam: float) -> pd.Series:
    """sigma^2_t = lam * sigma^2_{t-1} + (1 - lam) * r^2_t  (RiskMetrics recursion).

    Element *t* of the result is the variance estimate that incorporates the
    return of day *t*, i.e. the **one-step-ahead forecast for day t+1**.
    """
    if not 0 < lam < 1:
        raise ValueError("lam must lie strictly between 0 and 1")
    return _as_series(returns).pow(2).ewm(alpha=1 - lam, adjust=False).mean()


def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Conditional volatility *forecast for each day*, using data up to t-1 only.

    The one-period shift is not cosmetic: without it every risk number below
    would peek at the return it is trying to forecast, and the VaR backtest
    would report a model that is impossible to run in production.
    """
    return np.sqrt(_ewma_variance(returns, lam).shift(1).bfill())


def ewma_forecast_volatility(returns: pd.Series, lam: float = 0.94) -> float:
    """One-step-ahead conditional volatility given the full history supplied."""
    var = _ewma_variance(returns, lam)
    return float(np.sqrt(var.iloc[-1])) if len(var) else np.nan


def rolling_volatility(returns: pd.Series, window: int = 252, periods: int = 252) -> pd.Series:
    return _as_series(returns).rolling(window).std(ddof=1) * np.sqrt(periods)


def rolling_beta(returns: pd.Series, benchmark: pd.Series, window: int = 252) -> pd.Series:
    joined = pd.concat([_as_series(returns), _as_series(benchmark)], axis=1).dropna()
    joined.columns = ["r", "b"]
    cov = joined["r"].rolling(window).cov(joined["b"])
    var = joined["b"].rolling(window).var(ddof=1)
    return (cov / var).replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------------
# Value at Risk / Expected Shortfall
# ---------------------------------------------------------------------------
def value_at_risk(
    returns: pd.Series, confidence: float = 0.99, method: str = "historical",
    lam: float = 0.94,
) -> float:
    """One-period VaR as a positive loss magnitude."""
    if method not in VAR_METHODS:
        raise ValueError(f"method must be one of {VAR_METHODS}, got {method!r}")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must lie in (0.5, 1.0)")

    r = _as_series(returns)
    if len(r) < 30:
        return np.nan
    alpha = 1.0 - confidence
    z = stats.norm.ppf(alpha)

    if method == "historical":
        return float(-np.quantile(r, alpha))

    if method == "gaussian":
        return float(-(r.mean() + r.std(ddof=1) * z))

    if method == "cornish_fisher":
        s = float(np.clip(stats.skew(r, bias=False), -_CF_MAX_SKEW, _CF_MAX_SKEW))
        k = float(np.clip(stats.kurtosis(r, bias=False), 0.0, _CF_MAX_EXCESS_KURT))
        z_cf = (
            z
            + (z**2 - 1) * s / 6.0
            + (z**3 - 3 * z) * k / 24.0
            - (2 * z**3 - 5 * z) * s**2 / 36.0
        )
        return float(-(r.mean() + r.std(ddof=1) * z_cf))

    if method == "ewma":
        # RiskMetrics sets the conditional mean to zero and uses the one-step-
        # ahead variance forecast conditioned on the whole history given.
        return float(-ewma_forecast_volatility(r, lam) * z)

    # filtered_historical — devolatilise, take the empirical tail of the
    # standardised residuals, then re-inflate with today's conditional vol.
    sigma_path = ewma_volatility(r, lam)
    standardised = (r / sigma_path.replace(0.0, np.nan)).dropna()
    if len(standardised) < 30:
        return np.nan
    sigma_next = ewma_forecast_volatility(r, lam)
    return float(-np.quantile(standardised, alpha) * sigma_next)


def expected_shortfall(
    returns: pd.Series, confidence: float = 0.975, method: str = "historical"
) -> float:
    """Average loss conditional on breaching VaR (a.k.a. CVaR / FRTB ES)."""
    r = _as_series(returns)
    if len(r) < 30:
        return np.nan
    alpha = 1.0 - confidence

    if method == "gaussian":
        z = stats.norm.ppf(alpha)
        return float(-(r.mean() - r.std(ddof=1) * stats.norm.pdf(z) / alpha))

    cutoff = np.quantile(r, alpha)
    tail = r[r <= cutoff]
    return float(-tail.mean()) if len(tail) else np.nan


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VarBacktest:
    """Out-of-sample coverage test of a rolling VaR model."""

    method: str
    confidence: float
    observations: int
    exceptions: int
    expected_exceptions: float
    exception_rate: float
    kupiec_lr: float
    kupiec_p: float
    christoffersen_lr: float
    christoffersen_p: float
    conditional_coverage_lr: float
    conditional_coverage_p: float

    @property
    def verdict(self) -> str:
        """Plain-English read of the two tests at the 5% level."""
        if not np.isfinite(self.kupiec_p):
            return "inconclusive"
        coverage_ok = self.kupiec_p > 0.05
        independence_ok = (
            self.christoffersen_p > 0.05 if np.isfinite(self.christoffersen_p) else True
        )
        if coverage_ok and independence_ok:
            return "pass"
        if not coverage_ok and self.exception_rate > (1 - self.confidence):
            return "fail — understates risk"
        if not coverage_ok:
            return "fail — overstates risk"
        return "fail — exceptions cluster"

    def to_dict(self) -> dict[str, object]:
        return {**self.__dict__, "verdict": self.verdict}


def _safe_log(x: float) -> float:
    return float(np.log(x)) if x > 0 else 0.0


def _kupiec(n: int, x: int, p: float) -> tuple[float, float]:
    """Unconditional coverage: is the exception *rate* right?"""
    if n == 0:
        return np.nan, np.nan
    pi_hat = x / n
    ll_null = (n - x) * _safe_log(1 - p) + x * _safe_log(p)
    ll_alt = (n - x) * _safe_log(1 - pi_hat) + x * _safe_log(pi_hat)
    lr = -2.0 * (ll_null - ll_alt)
    lr = max(lr, 0.0)
    return float(lr), float(1.0 - stats.chi2.cdf(lr, df=1))


def _christoffersen(hits: np.ndarray) -> tuple[float, float]:
    """Independence: do exceptions cluster (i.e. does the model miss regimes)?"""
    if len(hits) < 2:
        return np.nan, np.nan
    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return np.nan, np.nan

    pi01 = n01 / (n00 + n01)
    pi11 = n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = (n00 + n10) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (
        n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01)
        + n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11)
    )
    lr = max(-2.0 * (ll_null - ll_alt), 0.0)
    return float(lr), float(1.0 - stats.chi2.cdf(lr, df=1))


def rolling_var_forecast(
    returns: pd.Series,
    confidence: float = 0.99,
    method: str = "filtered_historical",
    window: int = 500,
    lam: float = 0.94,
) -> pd.Series:
    """The out-of-sample VaR path the backtest actually evaluates.

    Element *t* is the forecast made with data up to *t-1*, so plotting it
    against realised returns shows exactly what the risk manager saw.
    """
    r = _as_series(returns)
    values = r.to_numpy()
    forecasts = np.full(len(values), np.nan)
    for t in range(window, len(values)):
        forecasts[t] = value_at_risk(pd.Series(values[t - window : t]), confidence, method, lam)
    return pd.Series(forecasts, index=r.index, name=f"var_{method}").dropna()


def var_backtest(
    returns: pd.Series,
    confidence: float = 0.99,
    method: str = "historical",
    window: int = 500,
    lam: float = 0.94,
) -> VarBacktest:
    """Roll the VaR model forward and test its realised coverage.

    At each date the model is estimated on the trailing ``window`` observations
    and compared with the *next* day's return, so no forecast ever sees its own
    outcome.
    """
    r = _as_series(returns)
    if len(r) <= window + 30:
        raise ValueError(f"Need more than {window + 30} observations, got {len(r)}")

    values = r.to_numpy()
    forecasts = np.full(len(values), np.nan)
    for t in range(window, len(values)):
        history = pd.Series(values[t - window : t])
        forecasts[t] = value_at_risk(history, confidence, method, lam)

    valid = ~np.isnan(forecasts)
    realised, var_hat = values[valid], forecasts[valid]
    hits = (realised < -var_hat).astype(int)

    n, x = len(hits), int(hits.sum())
    p = 1.0 - confidence
    kup_lr, kup_p = _kupiec(n, x, p)
    chr_lr, chr_p = _christoffersen(hits)

    cc_lr = kup_lr + chr_lr if np.isfinite(chr_lr) else np.nan
    cc_p = float(1.0 - stats.chi2.cdf(cc_lr, df=2)) if np.isfinite(cc_lr) else np.nan

    return VarBacktest(
        method=method,
        confidence=confidence,
        observations=n,
        exceptions=x,
        expected_exceptions=n * p,
        exception_rate=x / n if n else np.nan,
        kupiec_lr=kup_lr,
        kupiec_p=kup_p,
        christoffersen_lr=chr_lr,
        christoffersen_p=chr_p,
        conditional_coverage_lr=cc_lr,
        conditional_coverage_p=cc_p,
    )


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def risk_summary(
    returns: pd.Series | pd.DataFrame,
    var_confidence: float = 0.99,
    es_confidence: float = 0.975,
    lam: float = 0.94,
    periods: int = 252,
) -> pd.DataFrame:
    """VaR / ES across all four estimators, one column per series."""
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    vc = f"{var_confidence * 100:g}"
    ec = f"{es_confidence * 100:g}"
    out: dict[str, dict[str, float]] = {}

    for col in frame.columns:
        r = _as_series(frame[col])
        out[str(col)] = {
            "Ann. volatility": float(r.std(ddof=1) * np.sqrt(periods)),
            f"VaR {vc}% historical": value_at_risk(r, var_confidence, "historical"),
            f"VaR {vc}% gaussian": value_at_risk(r, var_confidence, "gaussian"),
            f"VaR {vc}% Cornish-Fisher": value_at_risk(r, var_confidence, "cornish_fisher"),
            f"VaR {vc}% EWMA (today)": value_at_risk(r, var_confidence, "ewma", lam),
            f"VaR {vc}% Filtered-HS (today)": value_at_risk(
                r, var_confidence, "filtered_historical", lam
            ),
            f"ES {ec}% historical": expected_shortfall(r, es_confidence, "historical"),
            f"ES {ec}% gaussian": expected_shortfall(r, es_confidence, "gaussian"),
            "Worst day": float(r.min()),
        }
    return pd.DataFrame(out)


def var_backtest_table(
    returns: pd.Series, confidence: float = 0.99, window: int = 500, lam: float = 0.94
) -> pd.DataFrame:
    """Run every estimator through the same coverage test and tabulate."""
    rows = []
    for method in VAR_METHODS:
        try:
            rows.append(var_backtest(returns, confidence, method, window, lam).to_dict())
        except ValueError:
            continue
    return pd.DataFrame(rows).set_index("method") if rows else pd.DataFrame()
