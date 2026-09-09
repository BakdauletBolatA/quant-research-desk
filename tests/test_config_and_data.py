"""Configuration integrity and the offline data path.

These tests run against the committed data cache, with no network access, and
are the reason the repository can be cloned and reproduced on a plane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantdesk.config import RAW_DIR, load_config, load_fundamentals
from quantdesk.data import build_panel

CACHE_READY = (RAW_DIR / "factors" / "ff5_mom_daily.csv").exists() and any(
    (RAW_DIR / "prices").glob("*.csv")
)
requires_cache = pytest.mark.skipif(not CACHE_READY, reason="data cache not populated")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_config_loads_and_is_internally_consistent():
    cfg = load_config()
    assert len(cfg.tickers) >= 8
    assert cfg.benchmark not in cfg.tickers, "the benchmark must sit outside the universe"
    assert cfg.all_symbols[-1] == cfg.benchmark
    assert len(set(cfg.all_symbols)) == len(cfg.all_symbols)
    assert set(cfg.sectors) == set(cfg.tickers)
    assert cfg.periods_per_year == 252
    assert pd.Timestamp(cfg.start) < pd.Timestamp(cfg.end)


def test_concentration_cap_is_feasible():
    cfg = load_config()
    portfolio = cfg["portfolio"]
    assert portfolio["max_weight"] * len(cfg.tickers) >= 1.0


def test_every_backtest_strategy_exists():
    from quantdesk.backtest.strategies import REGISTRY

    for name in load_config()["backtest"]["strategies"]:
        assert name in REGISTRY, name


def test_valuation_inputs_cover_only_universe_names():
    cfg = load_config()
    fundamentals = load_fundamentals()
    universe = set(cfg.tickers)
    assert set(fundamentals["dcf"]) <= universe
    assert set(fundamentals["excess_return"]) <= universe
    assert universe <= set(fundamentals["shares_outstanding"])
    # Banks must not appear in the FCFF set — that is the whole point of the split.
    assert not set(fundamentals["dcf"]) & set(fundamentals["excess_return"])


@pytest.mark.parametrize("ticker", list(load_fundamentals()["dcf"]))
def test_dcf_inputs_are_economically_sane(ticker):
    inputs = load_fundamentals()["dcf"][ticker]
    required = (
        "revenue", "ebit", "d_and_a", "capex", "delta_nwc", "tax_rate", "total_debt",
        "cash_and_investments", "shares_diluted", "beta", "cost_of_debt",
        "revenue_growth", "ebit_margin", "terminal_growth", "terminal_roic",
    )
    for field in required:
        assert field in inputs, f"{ticker} missing {field}"

    assert inputs["revenue"] > 0 and inputs["ebit"] > 0
    assert 0 < inputs["tax_rate"] < 0.6
    assert inputs["shares_diluted"] > 0
    assert 0 < inputs["beta"] < 3
    assert len(inputs["revenue_growth"]) == len(inputs["ebit_margin"])
    assert all(0 < m < 1 for m in inputs["ebit_margin"])
    assert 0 <= inputs["terminal_growth"] < 0.05, "terminal growth above nominal GDP"
    assert inputs["terminal_roic"] > inputs["terminal_growth"]


@pytest.mark.parametrize("ticker", list(load_fundamentals()["excess_return"]))
def test_bank_inputs_are_economically_sane(ticker):
    inputs = load_fundamentals()["excess_return"][ticker]
    assert inputs["book_value_equity"] > 0
    assert inputs["shares_diluted"] > 0
    assert 0 <= inputs["payout_ratio"] <= 1
    assert all(0 < roe < 0.5 for roe in inputs["roe_path"])
    assert 0 <= inputs["terminal_growth"] < 0.05


def test_black_litterman_views_reference_real_tickers():
    cfg = load_config()
    universe = set(cfg.tickers)
    for view in cfg["portfolio"]["black_litterman"]["views"]:
        assert view["type"] in ("absolute", "relative")
        assert set(view["assets"]) <= universe, view
        assert 0 < view["confidence"] < 1
        if view["type"] == "relative":
            assert len(view["assets"]) >= 2


# ---------------------------------------------------------------------------
# Offline data path
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel():
    if not CACHE_READY:
        pytest.skip("data cache not populated")
    return build_panel(load_config())


@requires_cache
def test_panel_is_complete_and_aligned(panel):
    assert not panel.returns.isna().to_numpy().any()
    assert not panel.prices.isna().to_numpy().any()
    assert panel.returns.index.is_monotonic_increasing
    assert panel.returns.index.is_unique
    assert panel.prices.index.equals(panel.benchmark_prices.index)
    assert panel.returns.index.equals(panel.benchmark_returns.index)
    assert panel.risk_free.index.equals(panel.returns.index)


@requires_cache
def test_prices_are_positive_and_returns_are_plausible(panel):
    assert (panel.prices > 0).to_numpy().all()
    assert panel.returns.abs().to_numpy().max() < 0.6, "a >60% daily move suggests bad data"


@requires_cache
def test_risk_free_is_carried_forward_not_dropped(panel):
    assert not panel.risk_free.isna().any()
    annualised = float(panel.risk_free.mean()) * panel.periods_per_year
    assert 0.0 <= annualised < 0.10


@requires_cache
def test_excess_returns_are_returns_minus_the_risk_free_path(panel):
    manual = panel.returns.sub(panel.risk_free, axis=0)
    assert np.allclose(panel.excess_returns.to_numpy(), manual.to_numpy(), atol=1e-15)


@requires_cache
def test_factor_panel_overlaps_the_price_calendar(panel):
    overlap = int(panel.factors["mkt_rf"].notna().sum())
    assert overlap > 250
    assert overlap <= len(panel.returns)
    assert set(panel.factors.columns) == {"mkt_rf", "smb", "hml", "rmw", "cma", "mom"}


@requires_cache
def test_summary_table_covers_the_universe(panel):
    summary = panel.summary()
    assert list(summary.index) == panel.tickers
    assert (summary["ann_vol"] > 0).all()
    assert summary["obs"].min() > 250


@requires_cache
def test_panel_is_deterministic():
    first, second = build_panel(load_config()), build_panel(load_config())
    assert first.returns.equals(second.returns)
