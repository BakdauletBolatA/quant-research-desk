"""Factor regression: does it recover a loading we planted?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from quantdesk.analytics.factor_model import (
    CAPM,
    FF5,
    FF5_MOM,
    factor_regression,
    factor_table,
    return_attribution,
)

TRUE_ALPHA_DAILY = 0.00020  # ~5% a year
TRUE_BETAS = {"mkt_rf": 1.10, "smb": -0.30, "hml": 0.20, "rmw": 0.15, "cma": -0.05,
              "mom": 0.10}


@pytest.fixture(scope="module")
def factors(dates) -> pd.DataFrame:
    generator = np.random.default_rng(2027)
    scale = {"mkt_rf": 0.010, "smb": 0.005, "hml": 0.005, "rmw": 0.004, "cma": 0.004,
             "mom": 0.006}
    return pd.DataFrame(
        {name: generator.normal(0.0002, sd, len(dates)) for name, sd in scale.items()},
        index=dates,
    )


@pytest.fixture(scope="module")
def planted_returns(factors) -> pd.Series:
    generator = np.random.default_rng(99)
    signal = sum(factors[name] * beta for name, beta in TRUE_BETAS.items())
    # Residual vol chosen so the planted alpha is clearly significant:
    # t ~ (alpha / sigma) * sqrt(n) = (0.0002 / 0.002) * sqrt(1500) ~ 3.9.
    noise = generator.normal(0.0, 0.002, len(factors))
    return (TRUE_ALPHA_DAILY + signal + noise).rename("planted")


def test_regression_recovers_the_planted_betas(planted_returns, factors):
    fit = factor_regression(planted_returns, factors, FF5_MOM, periods=252)
    for name, beta in TRUE_BETAS.items():
        assert fit.betas[name] == pytest.approx(beta, abs=0.06), name


def test_regression_recovers_the_planted_alpha(planted_returns, factors):
    fit = factor_regression(planted_returns, factors, FF5_MOM, periods=252)
    assert fit.alpha_annual == pytest.approx(TRUE_ALPHA_DAILY * 252, abs=0.015)
    assert fit.alpha_is_significant
    assert "survives factor controls" in fit.verdict


def test_a_pure_factor_portfolio_has_no_alpha(factors):
    """Factor exposure plus noise, no skill: alpha must not print as significant."""
    generator = np.random.default_rng(555)
    pure = (
        factors["mkt_rf"] * 1.2
        + factors["hml"] * 0.4
        + generator.normal(0.0, 0.003, len(factors))
    ).rename("pure")
    fit = factor_regression(pure, factors, FF5_MOM, periods=252)
    assert fit.alpha_annual == pytest.approx(0.0, abs=0.02)
    assert not fit.alpha_is_significant
    assert fit.betas["mkt_rf"] == pytest.approx(1.2, abs=0.05)
    assert "no reliable alpha" in fit.verdict


def test_a_noiseless_factor_replication_is_explained_completely(factors):
    exact = (factors["mkt_rf"] * 1.2 + factors["hml"] * 0.4).rename("exact")
    fit = factor_regression(exact, factors, FF5_MOM, periods=252)
    assert fit.r_squared > 0.999999
    assert fit.alpha_annual == pytest.approx(0.0, abs=1e-6)
    assert fit.residual_vol_annual < 1e-12


def test_newey_west_errors_differ_from_plain_ols(planted_returns, factors):
    """If HAC and OLS agreed there would be no reason to use HAC."""
    data = pd.concat([planted_returns.rename("y"), factors[FF5_MOM]], axis=1).dropna()
    X = sm.add_constant(data[FF5_MOM].to_numpy())
    ols = sm.OLS(data["y"].to_numpy(), X).fit()
    fit = factor_regression(planted_returns, factors, FF5_MOM, periods=252, nw_lags=5)
    assert fit.alpha_tstat != pytest.approx(float(ols.tvalues[0]), rel=1e-6)
    assert fit.betas["mkt_rf"] == pytest.approx(float(ols.params[1]), rel=1e-10)


def test_r_squared_rises_with_the_factor_set(planted_returns, factors):
    capm = factor_regression(planted_returns, factors, CAPM, periods=252)
    ff5 = factor_regression(planted_returns, factors, FF5, periods=252)
    full = factor_regression(planted_returns, factors, FF5_MOM, periods=252)
    assert capm.r_squared < ff5.r_squared < full.r_squared


def test_attribution_adds_up(planted_returns, factors):
    fit = factor_regression(planted_returns, factors, FF5_MOM, periods=252)
    attribution = return_attribution(fit, factors, periods=252)
    parts = attribution.drop("Total explained")
    assert attribution["Total explained"] == pytest.approx(parts.sum())
    assert attribution["Alpha"] == pytest.approx(fit.alpha_annual)
    realised = planted_returns.mean() * 252
    assert attribution["Total explained"] == pytest.approx(realised, abs=0.01)


def test_missing_factors_and_short_samples_raise(planted_returns, factors):
    with pytest.raises(ValueError, match="None of"):
        factor_regression(planted_returns, factors, ["not_a_factor"])
    with pytest.raises(ValueError, match="at least 60"):
        factor_regression(planted_returns.head(30), factors, CAPM)


def test_factor_table_covers_every_column(synthetic_returns, factors):
    table = factor_table(synthetic_returns, factors, FF5_MOM, periods=252)
    assert list(table.index) == list(synthetic_returns.columns)
    for column in ("Alpha (ann.)", "Alpha t-stat", "Market", "Adj. R²", "Obs"):
        assert column in table.columns


def test_residual_volatility_is_below_total_volatility(planted_returns, factors):
    fit = factor_regression(planted_returns, factors, FF5_MOM, periods=252)
    total = float(planted_returns.std(ddof=1) * np.sqrt(252))
    assert 0 < fit.residual_vol_annual < total
