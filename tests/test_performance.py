"""Performance statistics against closed-form answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdesk.analytics import performance as perf


def test_cagr_of_constant_return_is_that_return():
    daily = 0.0004
    series = pd.Series([daily] * 252, index=pd.bdate_range("2020-01-01", periods=252))
    expected = (1 + daily) ** 252 - 1
    assert perf.annualised_return(series, 252) == pytest.approx(expected, rel=1e-12)


def test_cagr_is_geometric_not_arithmetic():
    """+50% then -50% loses money; an arithmetic mean would report zero."""
    series = pd.Series([0.5, -0.5])
    assert perf.cumulative_return(series) == pytest.approx(-0.25)
    assert perf.annualised_return(series, 2) == pytest.approx(-0.25)


def test_annualised_volatility_scales_with_root_time():
    generator = np.random.default_rng(0)
    series = pd.Series(generator.normal(0, 0.01, 5000))
    assert perf.annualised_volatility(series, 252) == pytest.approx(
        series.std(ddof=1) * np.sqrt(252), rel=1e-12
    )


def test_sharpe_matches_manual_calculation(synthetic_series):
    risk_free = pd.Series(0.0001, index=synthetic_series.index)
    excess = synthetic_series - risk_free
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(252)
    assert perf.sharpe_ratio(synthetic_series, risk_free, 252) == pytest.approx(expected)


def test_sharpe_accepts_scalar_and_series_risk_free(synthetic_series):
    scalar = perf.sharpe_ratio(synthetic_series, 0.0001, 252)
    series = perf.sharpe_ratio(
        synthetic_series, pd.Series(0.0001, index=synthetic_series.index), 252
    )
    assert scalar == pytest.approx(series)


def test_sortino_exceeds_sharpe_for_right_skewed_returns():
    generator = np.random.default_rng(3)
    series = pd.Series(np.abs(generator.normal(0, 0.01, 2000)) - 0.004)
    assert perf.sortino_ratio(series, 0.0, 252) > perf.sharpe_ratio(series, 0.0, 252)


def test_max_drawdown_on_a_hand_built_path():
    # 1.0 -> 1.2 -> 0.6 -> 0.9 : peak 1.2, trough 0.6, drawdown -50%
    series = pd.Series([0.2, -0.5, 0.5])
    assert perf.max_drawdown(series) == pytest.approx(-0.5)


def test_drawdown_series_is_never_positive(synthetic_series):
    assert perf.drawdown_series(synthetic_series).max() <= 1e-12


def test_drawdown_table_identifies_the_deepest_episode():
    index = pd.bdate_range("2020-01-01", periods=7)
    series = pd.Series([0.0, 0.10, -0.40, -0.10, 0.60, 0.05, 0.0], index=index)
    table = perf.drawdown_table(series, top=3)
    assert not table.empty
    assert table["depth"].iloc[0] == pytest.approx(perf.max_drawdown(series))
    assert table["trough"].iloc[0] == index[3]


def test_calmar_is_cagr_over_drawdown(synthetic_series):
    expected = perf.annualised_return(synthetic_series, 252) / abs(
        perf.max_drawdown(synthetic_series)
    )
    assert perf.calmar_ratio(synthetic_series, 252) == pytest.approx(expected)


def test_beta_of_a_series_against_itself_is_one(synthetic_series):
    assert perf.beta(synthetic_series, synthetic_series) == pytest.approx(1.0)


@pytest.mark.parametrize("ticker,expected", [("HIGH", 1.20), ("MID", 1.00), ("LOW", 0.70)])
def test_beta_recovers_the_known_loading(synthetic_returns, synthetic_market, ticker, expected):
    assert perf.beta(synthetic_returns[ticker], synthetic_market) == pytest.approx(
        expected, abs=0.06
    )


def test_alpha_against_self_is_zero(synthetic_series):
    assert perf.jensen_alpha(synthetic_series, synthetic_series, 0.0, 252) == pytest.approx(
        0.0, abs=1e-12
    )


def test_information_ratio_of_a_perfect_tracker_is_nan(synthetic_series):
    assert np.isnan(perf.information_ratio(synthetic_series, synthetic_series, 252))


def test_psr_is_a_probability_and_rises_with_sample_length():
    generator = np.random.default_rng(11)
    short = pd.Series(generator.normal(0.0006, 0.01, 250))
    long = pd.Series(np.tile(short.to_numpy(), 8))
    p_short = perf.probabilistic_sharpe_ratio(short, 0.0, 252)
    p_long = perf.probabilistic_sharpe_ratio(long, 0.0, 252)
    assert 0.0 <= p_short <= 1.0 and 0.0 <= p_long <= 1.0
    assert p_long > p_short


def test_psr_falls_as_the_benchmark_sharpe_rises(synthetic_series):
    easy = perf.probabilistic_sharpe_ratio(synthetic_series, 0.0, 252)
    hard = perf.probabilistic_sharpe_ratio(synthetic_series, 2.0, 252)
    assert easy > hard


def test_expected_max_sharpe_grows_with_the_number_of_trials():
    variance = 1.0 / 999
    assert perf.expected_max_sharpe(100, variance) > perf.expected_max_sharpe(10, variance)
    assert perf.expected_max_sharpe(1, variance) == 0.0


def test_deflated_sharpe_is_no_greater_than_psr(synthetic_series):
    psr = perf.probabilistic_sharpe_ratio(synthetic_series, 0.0, 252)
    dsr = perf.deflated_sharpe_ratio(synthetic_series, n_trials=50, periods=252)
    assert dsr <= psr + 1e-12


def test_performance_summary_shape_and_labels(synthetic_returns):
    summary = perf.performance_summary(
        synthetic_returns, synthetic_returns["MID"], 0.0001, 252
    )
    assert list(summary.columns) == list(synthetic_returns.columns)
    for row in ("CAGR", "Sharpe", "Max drawdown", "Beta", "Alpha (ann.)"):
        assert row in summary.index
    assert summary.notna().to_numpy().sum() > 0


def test_empty_input_returns_nan_not_an_exception():
    empty = pd.Series(dtype=float)
    assert np.isnan(perf.annualised_return(empty))
    assert np.isnan(perf.sharpe_ratio(empty))
    assert np.isnan(perf.max_drawdown(empty))
