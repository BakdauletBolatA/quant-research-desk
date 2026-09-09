"""Backtest engine invariants.

The look-ahead test is the important one. Everything else in a backtest can be
approximately right and still be useful; a look-ahead bug makes every number
above it meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdesk.backtest.engine import _rebalance_calendar, run_all, run_backtest
from quantdesk.backtest.strategies import REGISTRY, align_weights
from quantdesk.portfolio.optimizer import Constraints

COMMON = {
    "start": "2018-01-01",
    "rebalance": "QE",
    "lookback_days": 504,
    "covariance_method": "ledoit_wolf",
    "periods_per_year": 252,
}


@pytest.fixture(scope="module")
def prices(synthetic_returns) -> pd.DataFrame:
    return (1 + synthetic_returns).cumprod() * 100.0


@pytest.fixture(scope="module")
def risk_free(synthetic_returns) -> pd.Series:
    return pd.Series(0.0001, index=synthetic_returns.index, name="rf")


@pytest.fixture(scope="module")
def shares() -> dict[str, float]:
    return {"HIGH": 1000.0, "MID": 800.0, "LOW": 600.0, "DEFENSIVE": 400.0}


def _run(name, synthetic_returns, prices, risk_free, shares, **overrides):
    kwargs = {
        **COMMON,
        "transaction_cost_bps": 10.0,
        "constraints": Constraints(True, 0.0, 0.60),
        "shares_outstanding": shares,
        "equity_risk_premium": 0.045,
        "views": [],
        "tau": 0.05,
        **overrides,
    }
    return run_backtest(name, synthetic_returns, prices, risk_free, **kwargs)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
def test_rebalance_dates_are_real_trading_days(synthetic_returns):
    dates = _rebalance_calendar(synthetic_returns.index, "QE")
    assert len(dates) > 8
    assert set(dates).issubset(set(synthetic_returns.index))
    assert dates == sorted(dates)


def test_monthly_rebalancing_produces_more_dates_than_quarterly(synthetic_returns):
    monthly = _rebalance_calendar(synthetic_returns.index, "ME")
    quarterly = _rebalance_calendar(synthetic_returns.index, "QE")
    assert len(monthly) > len(quarterly) * 2


# ---------------------------------------------------------------------------
# The look-ahead test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["min_variance", "risk_parity", "hrp", "max_sharpe"])
def test_no_look_ahead(synthetic_returns, prices, risk_free, shares, strategy):
    """Weights chosen on date d must not change when the future is deleted."""
    full = _run(strategy, synthetic_returns, prices, risk_free, shares)
    cutoff = full.rebalance_dates[-3]

    truncated = _run(
        strategy,
        synthetic_returns.loc[:cutoff],
        prices.loc[:cutoff],
        risk_free.loc[:cutoff],
        shares,
    )
    assert cutoff in truncated.weights.index
    assert np.allclose(
        full.weights.loc[cutoff].to_numpy(),
        truncated.weights.loc[cutoff].to_numpy(),
        atol=1e-10,
    ), strategy


def test_returns_never_use_the_same_day_weights(synthetic_returns, prices, risk_free, shares):
    """Doctoring a single future day must not change any earlier return."""
    baseline = _run("equal_weight", synthetic_returns, prices, risk_free, shares)
    tampered_returns = synthetic_returns.copy()
    shock_date = baseline.returns.index[-30]
    tampered_returns.loc[shock_date] = 0.5

    tampered = _run("equal_weight", tampered_returns, prices, risk_free, shares)
    before = baseline.returns.loc[: shock_date - pd.Timedelta(days=1)]
    assert np.allclose(before.to_numpy(),
                       tampered.returns.reindex(before.index).to_numpy(), atol=1e-12)


# ---------------------------------------------------------------------------
# Weights, drift and costs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", sorted(REGISTRY))
def test_every_strategy_produces_valid_weights(
    synthetic_returns, prices, risk_free, shares, strategy
):
    result = _run(strategy, synthetic_returns, prices, risk_free, shares)
    assert not result.weights.empty
    assert np.allclose(result.weights.sum(axis=1).to_numpy(), 1.0, atol=1e-8)
    assert (result.weights.to_numpy() >= -1e-9).all()
    assert result.weights.to_numpy().max() <= 0.60 + 1e-6
    assert len(result.returns) > 200


def test_zero_cost_makes_net_equal_gross(synthetic_returns, prices, risk_free, shares):
    result = _run("min_variance", synthetic_returns, prices, risk_free, shares,
                  transaction_cost_bps=0.0)
    assert np.allclose(result.returns.to_numpy(), result.gross_returns.to_numpy(), atol=1e-15)
    assert result.costs.abs().max() == pytest.approx(0.0)


def test_costs_only_ever_subtract(synthetic_returns, prices, risk_free, shares):
    result = _run("max_sharpe", synthetic_returns, prices, risk_free, shares,
                  transaction_cost_bps=25.0)
    assert (result.costs >= -1e-15).all()
    assert result.returns.sum() < result.gross_returns.sum()
    assert result.total_cost_drag > 0


def test_higher_costs_reduce_the_track_record(synthetic_returns, prices, risk_free, shares):
    cheap = _run("momentum_12_1", synthetic_returns, prices, risk_free, shares,
                 transaction_cost_bps=1.0)
    dear = _run("momentum_12_1", synthetic_returns, prices, risk_free, shares,
                transaction_cost_bps=100.0)
    assert dear.equity_curve.iloc[-1] < cheap.equity_curve.iloc[-1]
    assert dear.total_cost_drag > cheap.total_cost_drag


def test_equal_weight_turnover_is_only_the_drift(synthetic_returns, prices, risk_free, shares):
    """A 1/N book still trades: prices move it away from 1/N between rebalances."""
    result = _run("equal_weight", synthetic_returns, prices, risk_free, shares)
    assert 0.0 < result.annual_turnover < 1.5
    assert result.turnover.iloc[0] == pytest.approx(1.0)  # initial build from cash


def test_momentum_is_concentrated_and_trades_more(wide_returns, risk_free):
    """Needs a universe wider than the holding count, or the screen holds everything."""
    wide_prices = (1 + wide_returns).cumprod() * 100.0
    wide_shares = dict.fromkeys(wide_returns.columns, 1000.0)
    momentum = _run("momentum_12_1", wide_returns, wide_prices, risk_free, wide_shares)
    equal = _run("equal_weight", wide_returns, wide_prices, risk_free, wide_shares)
    assert momentum.average_n_positions == pytest.approx(5.0)
    assert equal.average_n_positions == pytest.approx(8.0)
    assert momentum.annual_turnover > equal.annual_turnover


def test_equity_curve_is_consistent_with_the_return_series(
    synthetic_returns, prices, risk_free, shares
):
    result = _run("risk_parity", synthetic_returns, prices, risk_free, shares)
    assert result.equity_curve.iloc[-1] == pytest.approx(
        float((1 + result.returns).prod())
    )


def test_unknown_strategy_and_short_window_raise(synthetic_returns, prices, risk_free, shares):
    with pytest.raises(ValueError):
        _run("does_not_exist", synthetic_returns, prices, risk_free, shares)
    with pytest.raises(ValueError, match="too short"):
        _run("equal_weight", synthetic_returns, prices, risk_free, shares, start="2020-12-01")


def test_run_all_shares_one_calendar(synthetic_returns, prices, risk_free, shares):
    results = run_all(
        ["equal_weight", "risk_parity", "hrp"], synthetic_returns, prices, risk_free,
        **COMMON, transaction_cost_bps=10.0, constraints=Constraints(True, 0.0, 0.60),
        shares_outstanding=shares, equity_risk_premium=0.045, views=[], tau=0.05,
    )
    assert set(results) == {"equal_weight", "risk_parity", "hrp"}
    indexes = [r.returns.index for r in results.values()]
    for index in indexes[1:]:
        assert index.equals(indexes[0])


def test_run_all_raises_only_when_everything_fails(
    synthetic_returns, prices, risk_free, shares
):
    with pytest.raises(RuntimeError):
        run_all(
            ["nope_1", "nope_2"], synthetic_returns, prices, risk_free, **COMMON,
            transaction_cost_bps=10.0, constraints=Constraints(),
            shares_outstanding=shares, equity_risk_premium=0.045, views=[], tau=0.05,
        )


def test_align_weights_renormalises_and_fills(synthetic_returns):
    columns = synthetic_returns.columns
    partial = pd.Series({"HIGH": 0.5, "MID": 0.5})
    aligned = align_weights(partial, columns)
    assert aligned.sum() == pytest.approx(1.0)
    assert len(aligned) == len(columns)
    assert aligned[-1] == pytest.approx(0.0)

    degenerate = align_weights(pd.Series({"HIGH": 0.0}), columns)
    assert np.allclose(degenerate, 1.0 / len(columns))
