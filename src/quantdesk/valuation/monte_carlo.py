"""Monte-Carlo uncertainty around the DCF drivers.

A single fair-value number implies a precision the inputs do not have. This
module perturbs the four drivers that actually move a DCF — revenue growth,
operating margin, WACC and terminal growth — and reports the *distribution* of
intrinsic value, together with the probability that the shares are undervalued
at the current price.

Implementation note: the projection is re-implemented here in vectorised NumPy
rather than looping ``run_dcf``. 20,000 paths through the pandas path takes
minutes; vectorised it takes milliseconds, and ``tests/test_valuation.py``
pins the two implementations to each other so the fast path cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MonteCarloResult:
    ticker: str
    current_price: float
    base_value: float
    values: np.ndarray = field(repr=False)
    n_paths: int = 0

    # -- distribution summary -------------------------------------------------
    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def median(self) -> float:
        return float(np.median(self.values))

    @property
    def std(self) -> float:
        return float(np.std(self.values, ddof=1))

    def percentile(self, q: float) -> float:
        return float(np.percentile(self.values, q))

    @property
    def prob_undervalued(self) -> float:
        """P(intrinsic value > market price) across the simulated inputs."""
        return float(np.mean(self.values > self.current_price))

    def summary(self) -> dict[str, float]:
        return {
            "Base case": self.base_value,
            "MC mean": self.mean,
            "MC median": self.median,
            "MC std dev": self.std,
            "P5": self.percentile(5),
            "P25": self.percentile(25),
            "P75": self.percentile(75),
            "P95": self.percentile(95),
            "Price": self.current_price,
            "P(undervalued)": self.prob_undervalued,
        }

    def to_series(self) -> pd.Series:
        return pd.Series(self.summary(), name=self.ticker)


def run_monte_carlo(
    ticker: str,
    inputs: dict,
    *,
    market: dict,
    current_price: float,
    base_wacc: float,
    base_value: float,
    mc_config: dict,
    n_paths: int = 20_000,
    seed: int = 42,
    terminal_roic_spread: float = 0.02,
    mid_year: bool = True,
) -> MonteCarloResult:
    """Simulate intrinsic value per share under driver uncertainty."""
    rng = np.random.default_rng(seed)

    revenue0 = float(inputs["revenue"])
    growth = np.asarray(inputs["revenue_growth"], dtype=float)
    margin = np.asarray(inputs["ebit_margin"], dtype=float)
    tax = float(inputs["tax_rate"])
    shares = float(inputs["shares_diluted"])
    net_debt = float(inputs["total_debt"]) - float(inputs["cash_and_investments"])
    n_years = len(growth)

    da_intensity = float(inputs["d_and_a"]) / revenue0
    capex_intensity = float(inputs["capex"]) / revenue0
    base_increment = revenue0 * growth[0]
    nwc_intensity = float(inputs["delta_nwc"]) / base_increment if base_increment else 0.0

    # --- draw the shocks ---------------------------------------------------
    g_shock = rng.normal(0.0, float(mc_config["revenue_growth_shock_sd"]), size=(n_paths, 1))
    m_shock = rng.normal(0.0, float(mc_config["ebit_margin_shock_sd"]), size=(n_paths, 1))
    wacc = base_wacc + rng.normal(0.0, float(mc_config["wacc_shock_sd"]), size=n_paths)
    term_g = float(inputs["terminal_growth"]) + rng.normal(
        0.0, float(mc_config["terminal_growth_shock_sd"]), size=n_paths
    )

    # Economic guardrails: growth cannot go structurally negative for ever, the
    # margin cannot exceed 100% or go below zero, and terminal growth must stay
    # a safe distance below the discount rate or the Gordon formula detonates.
    growth_paths = np.clip(growth[None, :] + g_shock, -0.10, 1.00)
    margin_paths = np.clip(margin[None, :] + m_shock, 0.01, 0.95)
    wacc = np.clip(wacc, 0.03, 0.30)
    cap_spread = float(mc_config.get("terminal_growth_cap_spread", 0.01))
    term_g = np.clip(term_g, -0.01, wacc - cap_spread)

    # --- vectorised projection ---------------------------------------------
    revenue = revenue0 * np.cumprod(1.0 + growth_paths, axis=1)
    ebit = revenue * margin_paths
    nopat = ebit * (1.0 - tax)
    d_and_a = revenue * da_intensity
    capex = revenue * capex_intensity

    prev_revenue = np.concatenate(
        [np.full((n_paths, 1), revenue0), revenue[:, :-1]], axis=1
    )
    delta_nwc = (revenue - prev_revenue) * nwc_intensity
    fcff = nopat + d_and_a - capex - delta_nwc

    offset = 0.5 if mid_year else 0.0
    periods = np.arange(1, n_years + 1, dtype=float) - offset
    discount = 1.0 / (1.0 + wacc[:, None]) ** periods[None, :]
    pv_explicit = np.sum(fcff * discount, axis=1)

    terminal_roic = float(inputs.get('terminal_roic') or 0.0) or (wacc + terminal_roic_spread)
    reinvestment = np.clip(term_g / terminal_roic, 0.0, 0.95)
    terminal_fcff = nopat[:, -1] * (1.0 + term_g) * (1.0 - reinvestment)
    terminal_value = terminal_fcff / (wacc - term_g)
    pv_terminal = terminal_value / (1.0 + wacc) ** n_years

    equity_value = pv_explicit + pv_terminal - net_debt
    values = equity_value / shares

    # A negative equity value is economically meaningful (the equity is a call
    # option struck at the debt) but it is not a per-share fair value, so the
    # distribution is floored at zero rather than dropped, which would bias the
    # upper percentiles.
    values = np.maximum(values, 0.0)

    return MonteCarloResult(
        ticker=ticker,
        current_price=current_price,
        base_value=base_value,
        values=values,
        n_paths=n_paths,
    )
