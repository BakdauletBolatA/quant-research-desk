"""Performance statistics.

Every function takes *simple* (arithmetic) periodic returns and an optional
risk-free series of the same frequency. Two conventions are fixed here and used
everywhere else in the platform:

1.  Annualised return is geometric (CAGR). Arithmetic annualisation flatters
    volatile assets and is the most common way a backtest lies to its author.
2.  Sharpe-family ratios use the arithmetic mean of *excess* returns over the
    realised risk-free path, not over a constant assumed rate.

The module also implements the Probabilistic and Deflated Sharpe Ratios
(Bailey & López de Prado). A Sharpe ratio without a confidence statement is a
point estimate from a short, fat-tailed, autocorrelated sample — which is to
say, close to meaningless on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_series(returns: pd.Series | np.ndarray) -> pd.Series:
    series = returns if isinstance(returns, pd.Series) else pd.Series(returns)
    return series.astype(float).dropna()


def _excess(returns: pd.Series, risk_free: pd.Series | float | None) -> pd.Series:
    if risk_free is None:
        return returns
    if isinstance(risk_free, (int, float)):
        return returns - float(risk_free)
    return (returns - risk_free.reindex(returns.index).ffill().fillna(0.0)).dropna()


# ---------------------------------------------------------------------------
# Core statistics
# ---------------------------------------------------------------------------
def cumulative_return(returns: pd.Series) -> float:
    """Total compounded return over the sample."""
    r = _as_series(returns)
    return float((1.0 + r).prod() - 1.0) if len(r) else np.nan


def annualised_return(returns: pd.Series, periods: int = 252) -> float:
    """Geometric annualised return (CAGR)."""
    r = _as_series(returns)
    if len(r) == 0:
        return np.nan
    growth = float((1.0 + r).prod())
    if growth <= 0:  # a total wipe-out has no finite CAGR
        return -1.0
    return growth ** (periods / len(r)) - 1.0


def annualised_volatility(returns: pd.Series, periods: int = 252) -> float:
    r = _as_series(returns)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else np.nan


def sharpe_ratio(
    returns: pd.Series, risk_free: pd.Series | float | None = None, periods: int = 252
) -> float:
    ex = _excess(_as_series(returns), risk_free)
    sd = ex.std(ddof=1)
    if len(ex) < 2 or sd == 0:
        return np.nan
    return float(ex.mean() / sd * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series,
    risk_free: pd.Series | float | None = None,
    periods: int = 252,
    target: float = 0.0,
) -> float:
    """Sharpe with the denominator restricted to downside deviation."""
    ex = _excess(_as_series(returns), risk_free)
    downside = np.minimum(ex - target, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if len(ex) < 2 or dd == 0:
        return np.nan
    return float(ex.mean() / dd * np.sqrt(periods))


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown path from the running peak of the compounded wealth index."""
    r = _as_series(returns)
    wealth = (1.0 + r).cumprod()
    return wealth / wealth.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    dd = drawdown_series(returns)
    return float(dd.min()) if len(dd) else np.nan


def drawdown_table(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """The ``top`` deepest peak-to-trough episodes with their recovery dates."""
    dd = drawdown_series(returns)
    if dd.empty:
        return pd.DataFrame(columns=["start", "trough", "recovery", "depth", "length_days"])

    underwater = dd < -1e-12
    episodes: list[dict[str, object]] = []
    start: pd.Timestamp | None = None

    for date, is_under in underwater.items():
        if is_under and start is None:
            start = date
        elif not is_under and start is not None:
            window = dd.loc[start:date]
            episodes.append({"start": start, "trough": window.idxmin(),
                             "recovery": date, "depth": float(window.min())})
            start = None
    if start is not None:  # still under water at the end of the sample
        window = dd.loc[start:]
        episodes.append({"start": start, "trough": window.idxmin(),
                         "recovery": pd.NaT, "depth": float(window.min())})

    table = pd.DataFrame(episodes)
    if table.empty:
        return pd.DataFrame(columns=["start", "trough", "recovery", "depth", "length_days"])
    table["length_days"] = (
        table["recovery"].fillna(dd.index[-1]) - table["start"]
    ).dt.days
    return (table.sort_values("depth").head(top).reset_index(drop=True))


def calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return np.nan
    return float(annualised_return(returns, periods) / abs(mdd))


def ulcer_index(returns: pd.Series) -> float:
    """RMS drawdown — penalises long, deep underwater periods, not just the worst day."""
    dd = drawdown_series(returns)
    return float(np.sqrt(np.mean(dd**2))) if len(dd) else np.nan


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Probability-weighted gains over losses relative to a threshold."""
    r = _as_series(returns) - threshold
    losses = -r[r < 0].sum()
    return float(r[r > 0].sum() / losses) if losses > 0 else np.nan


def tail_ratio(returns: pd.Series) -> float:
    """95th percentile gain divided by the absolute 5th percentile loss."""
    r = _as_series(returns)
    left = abs(np.percentile(r, 5))
    return float(np.percentile(r, 95) / left) if left > 0 else np.nan


def hit_rate(returns: pd.Series) -> float:
    r = _as_series(returns)
    return float((r > 0).mean()) if len(r) else np.nan


# ---------------------------------------------------------------------------
# Benchmark-relative statistics
# ---------------------------------------------------------------------------
def beta(returns: pd.Series, benchmark: pd.Series) -> float:
    joined = pd.concat([_as_series(returns), _as_series(benchmark)], axis=1).dropna()
    if len(joined) < 2:
        return np.nan
    r, b = joined.iloc[:, 0], joined.iloc[:, 1]
    var_b = float(b.var(ddof=1))
    return float(r.cov(b) / var_b) if var_b > 0 else np.nan


def jensen_alpha(
    returns: pd.Series,
    benchmark: pd.Series,
    risk_free: pd.Series | float | None = None,
    periods: int = 252,
) -> float:
    """Annualised CAPM alpha against the benchmark."""
    ex_r = _excess(_as_series(returns), risk_free)
    ex_b = _excess(_as_series(benchmark), risk_free)
    joined = pd.concat([ex_r, ex_b], axis=1).dropna()
    if len(joined) < 2:
        return np.nan
    b = beta(joined.iloc[:, 0], joined.iloc[:, 1])
    return float((joined.iloc[:, 0].mean() - b * joined.iloc[:, 1].mean()) * periods)


def tracking_error(returns: pd.Series, benchmark: pd.Series, periods: int = 252) -> float:
    active = (_as_series(returns) - _as_series(benchmark)).dropna()
    return float(active.std(ddof=1) * np.sqrt(periods)) if len(active) > 1 else np.nan


def information_ratio(returns: pd.Series, benchmark: pd.Series, periods: int = 252) -> float:
    active = (_as_series(returns) - _as_series(benchmark)).dropna()
    sd = active.std(ddof=1)
    if len(active) < 2 or sd == 0:
        return np.nan
    return float(active.mean() / sd * np.sqrt(periods))


def capture_ratios(returns: pd.Series, benchmark: pd.Series) -> tuple[float, float]:
    """(upside capture, downside capture) versus the benchmark, in ratio terms."""
    joined = pd.concat([_as_series(returns), _as_series(benchmark)], axis=1).dropna()
    if joined.empty:
        return np.nan, np.nan
    r, b = joined.iloc[:, 0], joined.iloc[:, 1]

    def _capture(mask: pd.Series) -> float:
        if mask.sum() < 2:
            return np.nan
        denom = float(b[mask].mean())
        return float(r[mask].mean() / denom) if denom != 0 else np.nan

    return _capture(b > 0), _capture(b < 0)


# ---------------------------------------------------------------------------
# Sharpe ratio inference (Bailey & López de Prado)
# ---------------------------------------------------------------------------
def probabilistic_sharpe_ratio(
    returns: pd.Series, benchmark_sr: float = 0.0, periods: int = 252,
    risk_free: pd.Series | float | None = None,
) -> float:
    """P(true Sharpe > ``benchmark_sr``) given the sample's skew and kurtosis.

    ``benchmark_sr`` is annualised, as is the observed Sharpe; both are
    converted to per-period units internally, which is where the sample size
    enters. Non-normality is not a footnote here: negative skew and fat tails
    inflate the observed Sharpe and this correction removes that inflation.
    """
    ex = _excess(_as_series(returns), risk_free)
    n = len(ex)
    sd = ex.std(ddof=1)
    if n < 3 or sd == 0:
        return np.nan

    sr = float(ex.mean() / sd)                      # per period
    sr_star = float(benchmark_sr / np.sqrt(periods))
    skew = float(stats.skew(ex, bias=False))
    kurt = float(stats.kurtosis(ex, fisher=False, bias=False))

    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return np.nan
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(n - 1) / np.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum per-period Sharpe from ``n_trials`` independent trials.

    This is the multiple-testing correction: run enough backtests and a Sharpe
    of 1.0 is the *expected* best result from pure noise.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpes: np.ndarray | pd.Series | None = None,
    periods: int = 252,
    risk_free: pd.Series | float | None = None,
) -> float:
    """PSR measured against the Sharpe you would expect from ``n_trials`` of noise.

    ``trial_sharpes`` (annualised) supplies the cross-trial variance; if it is
    omitted the observed strategy's own sampling variance is used as a proxy.
    """
    ex = _excess(_as_series(returns), risk_free)
    if len(ex) < 3:
        return np.nan

    if trial_sharpes is not None and len(np.asarray(trial_sharpes)) > 1:
        sr_var = float(np.nanvar(np.asarray(trial_sharpes, dtype=float), ddof=1) / periods)
    else:
        sr_var = 1.0 / max(len(ex) - 1, 1)

    sr0_annual = expected_max_sharpe(n_trials, sr_var) * np.sqrt(periods)
    return probabilistic_sharpe_ratio(ex, benchmark_sr=sr0_annual, periods=periods)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
def performance_summary(
    returns: pd.Series | pd.DataFrame,
    benchmark: pd.Series | None = None,
    risk_free: pd.Series | float | None = None,
    periods: int = 252,
) -> pd.DataFrame:
    """One column per series, one row per statistic."""
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    out: dict[str, dict[str, float]] = {}

    for col in frame.columns:
        r = _as_series(frame[col])
        stats_row: dict[str, float] = {
            "Total return": cumulative_return(r),
            "CAGR": annualised_return(r, periods),
            "Ann. volatility": annualised_volatility(r, periods),
            "Sharpe": sharpe_ratio(r, risk_free, periods),
            "Sortino": sortino_ratio(r, risk_free, periods),
            "Calmar": calmar_ratio(r, periods),
            "Max drawdown": max_drawdown(r),
            "Ulcer index": ulcer_index(r),
            "Omega": omega_ratio(r),
            "Skew": float(stats.skew(r, bias=False)) if len(r) > 2 else np.nan,
            "Excess kurtosis": float(stats.kurtosis(r, bias=False)) if len(r) > 3 else np.nan,
            "Tail ratio": tail_ratio(r),
            "Hit rate": hit_rate(r),
            "Best day": float(r.max()) if len(r) else np.nan,
            "Worst day": float(r.min()) if len(r) else np.nan,
            "PSR (vs 0)": probabilistic_sharpe_ratio(r, 0.0, periods, risk_free),
        }
        if benchmark is not None:
            up, down = capture_ratios(r, benchmark)
            stats_row |= {
                "Beta": beta(r, benchmark),
                "Alpha (ann.)": jensen_alpha(r, benchmark, risk_free, periods),
                "Tracking error": tracking_error(r, benchmark, periods),
                "Information ratio": information_ratio(r, benchmark, periods),
                "Upside capture": up,
                "Downside capture": down,
            }
        out[str(col)] = stats_row

    return pd.DataFrame(out)
