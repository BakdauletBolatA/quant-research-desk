"""Fama-French factor attribution with heteroskedasticity- and
autocorrelation-consistent inference.

The question this module answers is the one a portfolio manager will actually
be asked in an interview: *is the outperformance alpha, or is it beta to a
factor you were not paid to take?*

Two details that are usually skipped and shouldn't be:

* **Newey-West standard errors.** Daily return residuals are heteroskedastic
  and autocorrelated. OLS standard errors on daily data routinely halve the
  true standard error of alpha, turning noise into a two-sigma "result".
* **Annualising the intercept, not the t-statistic.** Alpha is reported in
  annual percent for readability, but significance is judged on the raw daily
  intercept's HAC t-statistic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm

FF5_MOM = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
FF5 = ["mkt_rf", "smb", "hml", "rmw", "cma"]
CAPM = ["mkt_rf"]

FACTOR_LABELS = {
    "mkt_rf": "Market",
    "smb": "Size (SMB)",
    "hml": "Value (HML)",
    "rmw": "Profitability (RMW)",
    "cma": "Investment (CMA)",
    "mom": "Momentum (MOM)",
}


@dataclass(frozen=True)
class FactorRegression:
    """Result of regressing excess returns on a factor set."""

    name: str
    factors: list[str]
    alpha_daily: float
    alpha_annual: float
    alpha_tstat: float
    alpha_pvalue: float
    betas: dict[str, float]
    tstats: dict[str, float]
    pvalues: dict[str, float]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residual_vol_annual: float
    nw_lags: int = 5
    _extra: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def alpha_is_significant(self) -> bool:
        """Two-sided 5% test on the HAC t-statistic."""
        return bool(np.isfinite(self.alpha_pvalue) and self.alpha_pvalue < 0.05)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.alpha_tstat):
            return "inconclusive"
        if not self.alpha_is_significant:
            return "no reliable alpha — returns are explained by factor exposure"
        return (
            f"alpha {self.alpha_annual:+.2%} p.a. survives factor controls "
            f"(t = {self.alpha_tstat:.2f})"
        )

    def to_row(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {
            "Alpha (ann.)": self.alpha_annual,
            "Alpha t-stat": self.alpha_tstat,
            "Alpha p-value": self.alpha_pvalue,
        }
        for factor in self.factors:
            row[FACTOR_LABELS.get(factor, factor)] = self.betas.get(factor, np.nan)
        row["R²"] = self.r_squared
        row["Adj. R²"] = self.adj_r_squared
        row["Resid. vol (ann.)"] = self.residual_vol_annual
        row["Obs"] = self.n_obs
        return row


def factor_regression(
    excess_returns: pd.Series,
    factors: pd.DataFrame,
    factor_names: list[str] | None = None,
    periods: int = 252,
    nw_lags: int = 5,
    name: str = "portfolio",
) -> FactorRegression:
    """Regress excess returns on the factor set with Newey-West (HAC) errors.

    ``excess_returns`` must already be net of the risk-free rate — the factors
    are excess-return series themselves, so mixing conventions here is the
    single most common way a factor regression is quietly wrong.
    """
    names = list(factor_names or FF5_MOM)
    available = [f for f in names if f in factors.columns]
    if not available:
        raise ValueError(f"None of {names} present in the factor panel")

    # Reindex onto the return series first: concatenating two differently
    # indexed frames leaves pandas to sort the union, which is both slower and
    # a deprecation warning waiting to happen.
    aligned = factors[available].reindex(excess_returns.index)
    data = pd.concat([excess_returns.rename("y"), aligned], axis=1).dropna()
    if len(data) < 60:
        raise ValueError(f"Need at least 60 aligned observations, got {len(data)}")

    y = data["y"].to_numpy()
    X = sm.add_constant(data[available].to_numpy(), has_constant="add")
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags, "use_correction": True})

    params, tvalues, pvalues = model.params, model.tvalues, model.pvalues
    residual_vol = float(np.std(model.resid, ddof=len(available) + 1) * np.sqrt(periods))

    return FactorRegression(
        name=name,
        factors=available,
        alpha_daily=float(params[0]),
        alpha_annual=float(params[0] * periods),
        alpha_tstat=float(tvalues[0]),
        alpha_pvalue=float(pvalues[0]),
        betas={f: float(params[i + 1]) for i, f in enumerate(available)},
        tstats={f: float(tvalues[i + 1]) for i, f in enumerate(available)},
        pvalues={f: float(pvalues[i + 1]) for i, f in enumerate(available)},
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        n_obs=int(model.nobs),
        residual_vol_annual=residual_vol,
        nw_lags=nw_lags,
    )


def factor_table(
    excess_returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_names: list[str] | None = None,
    periods: int = 252,
    nw_lags: int = 5,
) -> pd.DataFrame:
    """One factor regression per column, stacked into a comparison table."""
    rows: dict[str, dict[str, float | str]] = {}
    for col in excess_returns.columns:
        try:
            fit = factor_regression(
                excess_returns[col], factors, factor_names, periods, nw_lags, name=str(col)
            )
            rows[str(col)] = fit.to_row()
        except ValueError:
            continue
    return pd.DataFrame(rows).T if rows else pd.DataFrame()


def return_attribution(
    fit: FactorRegression, factors: pd.DataFrame, periods: int = 252
) -> pd.Series:
    """Decompose annualised excess return into factor contributions + alpha.

    contribution_k = beta_k * mean(factor_k) * periods
    """
    contributions = {
        FACTOR_LABELS.get(f, f): float(fit.betas[f] * factors[f].dropna().mean() * periods)
        for f in fit.factors
    }
    contributions["Alpha"] = fit.alpha_annual
    series = pd.Series(contributions)
    series["Total explained"] = float(series.sum())
    return series
