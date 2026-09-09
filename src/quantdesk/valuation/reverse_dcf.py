"""Reverse DCF — solving for what the market already believes.

A forward DCF answers "what do I think this is worth?". In a market where
mega-cap multiples sit far above the level a conservative model can justify,
that question produces a page of SELL ratings and very little insight.

The reverse DCF asks the more useful question: **what would I have to believe
to pay today's price?** It holds the valuation framework fixed, sets fair value
equal to the market price, and solves for the input:

* ``implied_terminal_growth``  — the perpetual growth rate embedded in the price.
  Anything above long-run nominal GDP (~4%) means the market expects the
  company to keep taking share of world output for ever.
* ``implied_revenue_cagr``     — the parallel shift to the explicit forecast
  needed to justify the price, reported as a 5-year revenue CAGR.
* ``implied_wacc``             — the discount rate that equates model value to
  price. This is the market's implied expected return on the equity+debt
  package: if it prints 6%, the market is accepting 6% for equity risk.

The output is falsifiable. "NVDA needs a 24% five-year revenue CAGR to justify
$224" is a claim a portfolio manager can argue with; "my DCF says fair value is
$77" is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize

from quantdesk.valuation.dcf import run_dcf

_SOLVER_XTOL = 1e-8
_MAX_ITER = 200


@dataclass(frozen=True)
class ReverseDCFResult:
    ticker: str
    current_price: float
    base_value: float
    base_wacc: float
    base_terminal_growth: float
    base_revenue_cagr: float
    implied_terminal_growth: float
    implied_revenue_cagr: float
    implied_wacc: float

    @property
    def terminal_growth_gap(self) -> float:
        return self.implied_terminal_growth - self.base_terminal_growth

    @property
    def revenue_cagr_gap(self) -> float:
        return self.implied_revenue_cagr - self.base_revenue_cagr

    def to_row(self) -> dict[str, float]:
        return {
            "Price": self.current_price,
            "Base fair value": self.base_value,
            "Base WACC": self.base_wacc,
            "Implied WACC": self.implied_wacc,
            "Base terminal g": self.base_terminal_growth,
            "Implied terminal g": self.implied_terminal_growth,
            "Base rev CAGR (5y)": self.base_revenue_cagr,
            "Implied rev CAGR (5y)": self.implied_revenue_cagr,
        }

    def narrative(self) -> str:
        """One sentence a portfolio manager can disagree with."""
        parts = []
        if np.isfinite(self.implied_revenue_cagr):
            parts.append(
                f"a {self.implied_revenue_cagr:.1%} five-year revenue CAGR "
                f"(base case {self.base_revenue_cagr:.1%})"
            )
        if np.isfinite(self.implied_wacc):
            parts.append(f"or accepting a {self.implied_wacc:.1%} cost of capital")
        if not parts:
            return f"{self.ticker}: no feasible set of inputs reproduces the market price."
        return f"To pay {self.current_price:,.2f} for {self.ticker} you must assume " + " ".join(
            parts
        ) + "."


def _revenue_cagr(growth_path: list[float]) -> float:
    """Compound annual growth implied by a per-year growth path."""
    arr = np.asarray(growth_path, dtype=float)
    return float(np.prod(1.0 + arr) ** (1.0 / len(arr)) - 1.0)


def _solve(func, lo: float, hi: float) -> float:
    """Brent root-find that returns NaN instead of raising when unbracketed."""
    try:
        f_lo, f_hi = func(lo), func(hi)
    except (ValueError, ZeroDivisionError):
        return np.nan
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return np.nan
    try:
        return float(
            optimize.brentq(func, lo, hi, xtol=_SOLVER_XTOL, maxiter=_MAX_ITER)
        )
    except (ValueError, RuntimeError):
        return np.nan


def reverse_dcf(
    ticker: str,
    inputs: dict,
    *,
    market: dict,
    current_price: float,
    base_result,
    **dcf_kwargs,
) -> ReverseDCFResult:
    """Back out the terminal growth, revenue CAGR and WACC implied by the price."""
    base_wacc = float(base_result.wacc)
    base_g = float(base_result.terminal_growth)
    base_growth_path = [float(x) for x in inputs["revenue_growth"]]
    base_cagr = _revenue_cagr(base_growth_path)
    coc = base_result.cost_of_capital

    def _value(**overrides) -> float:
        return run_dcf(
            ticker, {**inputs, **overrides.pop("inputs", {})}, market=market,
            current_price=current_price, cost_of_capital=coc, **overrides, **dcf_kwargs,
        ).value_per_share

    # --- implied terminal growth -------------------------------------------
    def f_growth(g: float) -> float:
        return _value(terminal_growth_override=g) - current_price

    implied_g = _solve(f_growth, -0.05, base_wacc - 0.005)

    # --- implied revenue CAGR (parallel shift of the explicit forecast) -----
    def f_shift(shift: float) -> float:
        shifted = [g + shift for g in base_growth_path]
        return _value(inputs={"revenue_growth": shifted}) - current_price

    shift = _solve(f_shift, -0.30, 0.60)
    implied_cagr = (
        _revenue_cagr([g + shift for g in base_growth_path]) if np.isfinite(shift) else np.nan
    )

    # --- implied WACC ------------------------------------------------------
    def f_wacc(w: float) -> float:
        return _value(wacc_override=w) - current_price

    implied_wacc = _solve(f_wacc, max(base_g + 0.006, 0.035), 0.35)

    return ReverseDCFResult(
        ticker=ticker,
        current_price=current_price,
        base_value=float(base_result.value_per_share),
        base_wacc=base_wacc,
        base_terminal_growth=base_g,
        base_revenue_cagr=base_cagr,
        implied_terminal_growth=implied_g,
        implied_revenue_cagr=implied_cagr,
        implied_wacc=implied_wacc,
    )
