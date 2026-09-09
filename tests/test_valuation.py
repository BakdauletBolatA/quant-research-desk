"""Valuation models against analytic identities.

The DCF is checked against a perpetuity it must reproduce exactly, the
vectorised Monte Carlo is pinned to the pandas reference implementation, and
the reverse DCF is checked by round-tripping its own solution.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantdesk.valuation import comps as comps_mod
from quantdesk.valuation.dcf import run_dcf, sensitivity_grid
from quantdesk.valuation.excess_return import run_excess_return_model
from quantdesk.valuation.monte_carlo import run_monte_carlo
from quantdesk.valuation.reverse_dcf import reverse_dcf
from quantdesk.valuation.wacc import (
    compute_wacc,
    cost_of_equity_capm,
    relever_beta,
    unlever_beta,
)


# ---------------------------------------------------------------------------
# Cost of capital
# ---------------------------------------------------------------------------
def test_capm_is_the_textbook_formula():
    assert cost_of_equity_capm(0.04, 1.2, 0.05) == pytest.approx(0.10)


def test_wacc_is_the_market_value_weighted_average():
    result = compute_wacc(
        market_cap=800.0, total_debt=200.0, beta=1.0, risk_free_rate=0.04,
        equity_risk_premium=0.05, cost_of_debt=0.06, tax_rate=0.25,
    )
    assert result.equity_weight == pytest.approx(0.80)
    assert result.cost_of_equity == pytest.approx(0.09)
    assert result.after_tax_cost_of_debt == pytest.approx(0.045)
    assert result.wacc == pytest.approx(0.80 * 0.09 + 0.20 * 0.045)


def test_all_equity_wacc_equals_cost_of_equity():
    result = compute_wacc(
        market_cap=1000.0, total_debt=0.0, beta=1.1, risk_free_rate=0.04,
        equity_risk_premium=0.05, cost_of_debt=0.06, tax_rate=0.25,
    )
    assert result.wacc == pytest.approx(result.cost_of_equity)


def test_hamada_unlever_relever_round_trip():
    levered = 1.35
    asset = unlever_beta(levered, debt_to_equity=0.5, tax_rate=0.25)
    assert asset < levered
    assert relever_beta(asset, 0.5, 0.25) == pytest.approx(levered)


def test_wacc_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        compute_wacc(market_cap=0.0, total_debt=100.0, beta=1.0, risk_free_rate=0.04,
                     equity_risk_premium=0.05, cost_of_debt=0.05, tax_rate=0.25)


# ---------------------------------------------------------------------------
# DCF
# ---------------------------------------------------------------------------
def test_zero_growth_dcf_equals_the_perpetuity_identity(dcf_inputs, market_assumptions):
    """capex = D&A, no growth, no working capital: EV must be exactly NOPAT / WACC."""
    inputs = {
        **dcf_inputs,
        "revenue_growth": [0.0] * 5,
        "capex": dcf_inputs["d_and_a"],
        "delta_nwc": 0.0,
        "terminal_growth": 0.0,
    }
    result = run_dcf(
        "TEST", inputs, market=market_assumptions, current_price=10.0,
        terminal_method="gordon_fcff", mid_year=False,
    )
    nopat = inputs["revenue"] * inputs["ebit_margin"][0] * (1 - inputs["tax_rate"])
    assert result.enterprise_value == pytest.approx(nopat / result.wacc, rel=1e-9)


def test_equity_value_is_enterprise_value_less_net_debt(dcf_inputs, market_assumptions):
    result = run_dcf("TEST", dcf_inputs, market=market_assumptions, current_price=10.0)
    net_debt = dcf_inputs["total_debt"] - dcf_inputs["cash_and_investments"]
    assert result.net_debt == pytest.approx(net_debt)
    assert result.equity_value == pytest.approx(result.enterprise_value - net_debt)
    assert result.value_per_share == pytest.approx(
        result.equity_value / dcf_inputs["shares_diluted"]
    )


def test_value_falls_as_wacc_rises(dcf_inputs, market_assumptions):
    low = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                  wacc_override=0.08)
    high = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                   wacc_override=0.12)
    assert low.value_per_share > high.value_per_share


def test_value_rises_with_terminal_growth(dcf_inputs, market_assumptions):
    low = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                  wacc_override=0.10, terminal_growth_override=0.01)
    high = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                   wacc_override=0.10, terminal_growth_override=0.03)
    assert high.value_per_share > low.value_per_share


def test_mid_year_discounting_is_worth_about_half_a_year(dcf_inputs, market_assumptions):
    end = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                  wacc_override=0.09, mid_year=False)
    mid = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                  wacc_override=0.09, mid_year=True)
    uplift = mid.pv_explicit / end.pv_explicit
    assert uplift == pytest.approx(np.sqrt(1.09), rel=1e-9)


def test_reinvestment_terminal_value_is_below_naive_gordon(dcf_inputs, market_assumptions):
    """Growth has to be paid for; ignoring that overstates the terminal value."""
    naive = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                    terminal_method="gordon_fcff")
    disciplined = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                          terminal_method="reinvestment")
    assert disciplined.terminal_value < naive.terminal_value


def test_wacc_must_exceed_terminal_growth(dcf_inputs, market_assumptions):
    with pytest.raises(ValueError, match="must exceed terminal growth"):
        run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                wacc_override=0.03, terminal_growth_override=0.05)


def test_terminal_roic_must_exceed_terminal_growth(dcf_inputs, market_assumptions):
    with pytest.raises(ValueError, match="terminal ROIC"):
        run_dcf("T", {**dcf_inputs, "terminal_roic": 0.01}, market=market_assumptions,
                current_price=10.0)


def test_projection_compounds_revenue_correctly(dcf_inputs, market_assumptions):
    result = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    projection = result.projection
    expected = dcf_inputs["revenue"] * np.cumprod(1 + np.array(dcf_inputs["revenue_growth"]))
    assert np.allclose(projection["revenue"].to_numpy(), expected)
    assert np.allclose(
        projection["nopat"].to_numpy(),
        projection["ebit"].to_numpy() * (1 - dcf_inputs["tax_rate"]),
    )


def test_terminal_value_share_is_a_fraction(dcf_inputs, market_assumptions):
    result = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    assert 0.0 < result.terminal_value_share < 1.0
    assert result.pv_explicit + result.pv_terminal_value == pytest.approx(
        result.enterprise_value
    )


def test_recommendation_bands(dcf_inputs, market_assumptions):
    base = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    fair = base.value_per_share
    assert run_dcf("T", dcf_inputs, market=market_assumptions,
                   current_price=fair * 0.5).recommendation == "BUY"
    assert run_dcf("T", dcf_inputs, market=market_assumptions,
                   current_price=fair).recommendation == "HOLD"
    assert run_dcf("T", dcf_inputs, market=market_assumptions,
                   current_price=fair * 2.0).recommendation == "SELL"


def test_sensitivity_grid_is_monotone_and_masks_infeasible_cells(
    dcf_inputs, market_assumptions
):
    base = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    grid = sensitivity_grid("T", dcf_inputs, market=market_assumptions, current_price=10.0,
                            base=base)
    assert grid.shape == (5, 5)
    for column in grid.columns:  # value falls as WACC rises
        values = grid[column].dropna()
        assert values.is_monotonic_decreasing
    for _, row in grid.iterrows():  # value rises with terminal growth
        values = row.dropna()
        assert values.is_monotonic_increasing


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def test_monte_carlo_with_zero_shocks_reproduces_the_dcf(
    dcf_inputs, market_assumptions, mc_config
):
    """Pins the fast vectorised path to the pandas reference implementation."""
    base = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    zero = dict.fromkeys(mc_config, 0.0)
    zero["terminal_growth_cap_spread"] = mc_config["terminal_growth_cap_spread"]
    result = run_monte_carlo(
        "T", dcf_inputs, market=market_assumptions, current_price=10.0,
        base_wacc=base.wacc, base_value=base.value_per_share, mc_config=zero,
        n_paths=64, seed=1,
    )
    assert result.values.std() == pytest.approx(0.0, abs=1e-9)
    assert result.mean == pytest.approx(base.value_per_share, rel=1e-10)


def test_monte_carlo_is_reproducible_and_bounded(dcf_inputs, market_assumptions, mc_config):
    base = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    kwargs = {
        "market": market_assumptions, "current_price": 10.0, "base_wacc": base.wacc,
        "base_value": base.value_per_share, "mc_config": mc_config, "n_paths": 4000,
    }
    first = run_monte_carlo("T", dcf_inputs, seed=42, **kwargs)
    second = run_monte_carlo("T", dcf_inputs, seed=42, **kwargs)
    third = run_monte_carlo("T", dcf_inputs, seed=43, **kwargs)

    assert np.array_equal(first.values, second.values)
    assert not np.array_equal(first.values, third.values)
    assert (first.values >= 0).all()
    assert first.percentile(5) < first.median < first.percentile(95)
    assert 0.0 <= first.prob_undervalued <= 1.0


def test_monte_carlo_dispersion_grows_with_input_uncertainty(
    dcf_inputs, market_assumptions, mc_config
):
    base = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    wide = {**mc_config, "wacc_shock_sd": mc_config["wacc_shock_sd"] * 3}
    kwargs = {
        "market": market_assumptions, "current_price": 10.0, "base_wacc": base.wacc,
        "base_value": base.value_per_share, "n_paths": 8000, "seed": 7,
    }
    tight = run_monte_carlo("T", dcf_inputs, mc_config=mc_config, **kwargs)
    loose = run_monte_carlo("T", dcf_inputs, mc_config=wide, **kwargs)
    assert loose.std > tight.std


# ---------------------------------------------------------------------------
# Reverse DCF
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("price_multiple", [0.6, 1.4])
def test_reverse_dcf_round_trips_its_own_solution(
    dcf_inputs, market_assumptions, price_multiple
):
    """Plugging each solved input back into the model must return the price."""
    anchor = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    price = anchor.value_per_share * price_multiple
    base = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=price)
    inverted = reverse_dcf("T", dcf_inputs, market=market_assumptions, current_price=price,
                           base_result=base)

    for field, kwargs in (
        ("implied_wacc", "wacc_override"),
        ("implied_terminal_growth", "terminal_growth_override"),
    ):
        solved = getattr(inverted, field)
        assert np.isfinite(solved), field
        recovered = run_dcf(
            "T", dcf_inputs, market=market_assumptions, current_price=price,
            cost_of_capital=base.cost_of_capital, **{kwargs: solved},
        )
        assert recovered.value_per_share == pytest.approx(price, rel=1e-6), field

    assert np.isfinite(inverted.implied_revenue_cagr)
    shifted = [
        g + (inverted.implied_revenue_cagr - inverted.base_revenue_cagr)
        for g in dcf_inputs["revenue_growth"]
    ]
    recovered_growth = run_dcf(
        "T", {**dcf_inputs, "revenue_growth": shifted}, market=market_assumptions,
        current_price=price, cost_of_capital=base.cost_of_capital,
    )
    assert recovered_growth.value_per_share == pytest.approx(price, rel=1e-4)


def test_a_price_above_fair_value_implies_a_lower_discount_rate(
    dcf_inputs, market_assumptions
):
    base = run_dcf("T", dcf_inputs, market=market_assumptions, current_price=10.0)
    expensive = base.value_per_share * 1.6
    inverted = reverse_dcf("T", dcf_inputs, market=market_assumptions,
                           current_price=expensive, base_result=base)
    assert inverted.implied_wacc < base.wacc
    assert inverted.implied_revenue_cagr > inverted.base_revenue_cagr
    assert "must assume" in inverted.narrative()


# ---------------------------------------------------------------------------
# Bank excess-return model
# ---------------------------------------------------------------------------
def test_a_bank_earning_its_cost_of_equity_is_worth_exactly_book(market_assumptions):
    inputs = {
        "book_value_equity": 1000.0,
        "shares_diluted": 100.0,
        "roe_path": [0.10] * 5,
        "payout_ratio": 0.5,
        "beta": 1.0,
        "terminal_roe": 0.10,
        "terminal_growth": 0.02,
    }
    result = run_excess_return_model(
        "BANK", inputs, market=market_assumptions, current_price=10.0,
        cost_of_equity_override=0.10,
    )
    assert result.value_per_share == pytest.approx(10.0, rel=1e-12)
    assert result.implied_p_b == pytest.approx(1.0, rel=1e-12)


def test_a_bank_out_earning_its_cost_of_equity_trades_above_book(market_assumptions):
    inputs = {
        "book_value_equity": 1000.0, "shares_diluted": 100.0, "roe_path": [0.16] * 5,
        "payout_ratio": 0.45, "beta": 1.0, "terminal_roe": 0.14, "terminal_growth": 0.02,
    }
    result = run_excess_return_model("BANK", inputs, market=market_assumptions,
                                     current_price=10.0, cost_of_equity_override=0.10)
    assert result.implied_p_b > 1.0
    assert result.projection["excess_return"].gt(0).all()


def test_bank_model_requires_cost_of_equity_above_growth(market_assumptions):
    inputs = {
        "book_value_equity": 1000.0, "shares_diluted": 100.0, "roe_path": [0.10] * 5,
        "payout_ratio": 0.5, "beta": 1.0, "terminal_roe": 0.10, "terminal_growth": 0.12,
    }
    with pytest.raises(ValueError):
        run_excess_return_model("BANK", inputs, market=market_assumptions,
                                current_price=10.0, cost_of_equity_override=0.10)


# ---------------------------------------------------------------------------
# Comparables
# ---------------------------------------------------------------------------
def test_multiples_are_computed_on_the_right_capital_base(dcf_inputs):
    table = comps_mod.comps_table({"T": dcf_inputs}, {"T": 10.0})
    row = table.loc["T"]
    market_cap = 10.0 * dcf_inputs["shares_diluted"]
    net_debt = dcf_inputs["total_debt"] - dcf_inputs["cash_and_investments"]
    assert row["EV"] == pytest.approx(market_cap + net_debt)
    assert row["EV/Sales"] == pytest.approx(row["EV"] / dcf_inputs["revenue"])
    assert row["EV/EBITDA"] == pytest.approx(
        row["EV"] / (dcf_inputs["ebit"] + dcf_inputs["d_and_a"])
    )


def test_implied_price_from_multiple_inverts_the_multiple(dcf_inputs):
    table = comps_mod.comps_table({"T": dcf_inputs}, {"T": 10.0})
    for basis in ("EV/Sales", "EV/EBITDA", "EV/EBIT", "P/E"):
        multiple = float(table.loc["T", basis])
        implied = comps_mod.implied_price_from_multiple(dcf_inputs, multiple, basis)
        assert implied == pytest.approx(10.0, rel=1e-9), basis


def test_unsupported_multiple_basis_raises(dcf_inputs):
    with pytest.raises(ValueError):
        comps_mod.implied_price_from_multiple(dcf_inputs, 10.0, "EV/NetIncome")


def test_football_field_brackets_every_methodology(dcf_inputs):
    comps = comps_mod.comps_table(
        {"A": dcf_inputs, "B": {**dcf_inputs, "ebit": 260.0},
         "C": {**dcf_inputs, "ebit": 150.0}},
        {"A": 10.0, "B": 12.0, "C": 8.0},
    )
    field = comps_mod.football_field(
        "A", dcf_inputs, comps, current_price=10.0, dcf_low=8.0, dcf_high=12.0,
        sensitivity_low=6.0, sensitivity_high=15.0, price_52w_low=7.0, price_52w_high=13.0,
    )
    assert (field["high"] >= field["low"]).all()
    assert np.allclose(field["mid"], (field["low"] + field["high"]) / 2)
    assert "52-week trading range" in field.index
