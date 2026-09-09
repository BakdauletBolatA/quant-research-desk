"""VaR / ES estimators and the model-validation machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantdesk.analytics import risk


def test_gaussian_var_is_the_closed_form():
    generator = np.random.default_rng(1)
    series = pd.Series(generator.normal(0.0005, 0.012, 4000))
    z = stats.norm.ppf(0.01)
    expected = -(series.mean() + series.std(ddof=1) * z)
    assert risk.value_at_risk(series, 0.99, "gaussian") == pytest.approx(expected, rel=1e-12)


def test_historical_var_is_the_empirical_quantile(synthetic_series):
    expected = -np.quantile(synthetic_series.to_numpy(), 0.01)
    assert risk.value_at_risk(synthetic_series, 0.99, "historical") == pytest.approx(expected)


def test_var_is_reported_as_a_positive_loss(synthetic_series):
    for method in risk.VAR_METHODS:
        assert risk.value_at_risk(synthetic_series, 0.99, method) > 0


def test_var_increases_with_confidence(synthetic_series):
    for method in ("historical", "gaussian", "cornish_fisher"):
        low = risk.value_at_risk(synthetic_series, 0.95, method)
        high = risk.value_at_risk(synthetic_series, 0.99, method)
        assert high > low, method


def test_expected_shortfall_is_at_least_var(synthetic_series):
    var = risk.value_at_risk(synthetic_series, 0.975, "historical")
    es = risk.expected_shortfall(synthetic_series, 0.975, "historical")
    assert es >= var


def test_gaussian_var_understates_a_fat_tailed_sample():
    """A Student-t sample has more tail than the normal model admits."""
    generator = np.random.default_rng(5)
    series = pd.Series(generator.standard_t(3, 6000) * 0.006)
    assert (
        risk.value_at_risk(series, 0.99, "historical")
        > risk.value_at_risk(series, 0.99, "gaussian")
    )


def test_cornish_fisher_moments_are_clamped_to_the_validity_domain():
    """Without the clamp an extreme-kurtosis sample returns a nonsense VaR."""
    generator = np.random.default_rng(9)
    series = pd.Series(np.concatenate([generator.normal(0, 0.008, 3000), [-0.35, 0.30]]))
    var = risk.value_at_risk(series, 0.99, "cornish_fisher")
    assert 0 < var < 0.25


def test_invalid_arguments_raise():
    series = pd.Series(np.zeros(100) + 0.001)
    with pytest.raises(ValueError):
        risk.value_at_risk(series, 0.99, "not_a_method")
    with pytest.raises(ValueError):
        risk.value_at_risk(series, 1.5, "historical")
    with pytest.raises(ValueError):
        risk.ewma_volatility(series, lam=1.0)


def test_ewma_forecast_never_sees_the_day_it_forecasts(synthetic_series):
    """The single most important property of the whole risk module."""
    baseline = risk.ewma_volatility(synthetic_series, 0.94)
    tampered = synthetic_series.copy()
    tampered.iloc[-1] = -0.5  # a crash on the final day
    after = risk.ewma_volatility(tampered, 0.94)
    assert after.iloc[-1] == pytest.approx(baseline.iloc[-1], rel=1e-12)
    # ...but the one-step-ahead forecast *does* react to it
    assert risk.ewma_forecast_volatility(tampered) > risk.ewma_forecast_volatility(
        synthetic_series
    )


def test_ewma_recursion_matches_a_manual_loop():
    generator = np.random.default_rng(4)
    values = generator.normal(0, 0.01, 500)
    lam = 0.94
    variance = values[0] ** 2
    for value in values[1:]:
        variance = lam * variance + (1 - lam) * value**2
    assert risk.ewma_forecast_volatility(pd.Series(values), lam) == pytest.approx(
        np.sqrt(variance), rel=1e-9
    )


def test_kupiec_is_zero_when_the_exception_rate_is_exactly_right():
    lr, p_value = risk._kupiec(n=1000, x=10, p=0.01)
    assert lr == pytest.approx(0.0, abs=1e-9)
    assert p_value == pytest.approx(1.0, abs=1e-9)


def test_kupiec_rejects_a_badly_calibrated_model():
    lr, p_value = risk._kupiec(n=1000, x=60, p=0.01)
    assert lr > 10 and p_value < 0.01


def test_christoffersen_rejects_clustered_exceptions():
    """Ten exceptions in a row is the same rate but the wrong pattern."""
    clustered = np.zeros(1000, dtype=int)
    clustered[500:510] = 1
    spread = np.zeros(1000, dtype=int)
    spread[::100] = 1
    _, p_clustered = risk._christoffersen(clustered)
    _, p_spread = risk._christoffersen(spread)
    assert p_clustered < 0.05
    assert p_spread > p_clustered


def test_var_backtest_is_well_calibrated_on_gaussian_data():
    """On truly normal data the gaussian model should pass its own test."""
    generator = np.random.default_rng(2026)
    series = pd.Series(generator.normal(0.0003, 0.011, 4000))
    result = risk.var_backtest(series, 0.99, "gaussian", window=500)
    assert result.observations == 3500
    assert result.exception_rate == pytest.approx(0.01, abs=0.006)
    assert result.kupiec_p > 0.05
    assert result.verdict == "pass"


def test_var_backtest_flags_a_model_that_understates_risk():
    """Gaussian VaR on fat-tailed data must be rejected."""
    generator = np.random.default_rng(31)
    series = pd.Series(generator.standard_t(3, 4000) * 0.006)
    result = risk.var_backtest(series, 0.99, "gaussian", window=500)
    assert result.exception_rate > 0.01
    assert result.kupiec_p < 0.05
    assert "understates" in result.verdict


def test_var_backtest_needs_enough_observations():
    with pytest.raises(ValueError):
        risk.var_backtest(pd.Series(np.zeros(400) + 0.001), window=500)


def test_rolling_var_forecast_is_aligned_and_shorter_than_the_input(synthetic_series):
    path = risk.rolling_var_forecast(synthetic_series, 0.99, "historical", window=500)
    assert len(path) == len(synthetic_series) - 500
    assert path.index[0] == synthetic_series.index[500]
    assert (path > 0).all()


def test_risk_summary_covers_every_estimator(synthetic_returns):
    summary = risk.risk_summary(synthetic_returns, 0.99, 0.975)
    assert list(summary.columns) == list(synthetic_returns.columns)
    assert any("Filtered-HS" in str(i) for i in summary.index)
    assert summary.notna().all().all()


def test_var_backtest_table_runs_all_methods(synthetic_series):
    table = risk.var_backtest_table(synthetic_series, 0.99, window=500)
    assert set(table.index) == set(risk.VAR_METHODS)
    assert (table["observations"] > 0).all()
