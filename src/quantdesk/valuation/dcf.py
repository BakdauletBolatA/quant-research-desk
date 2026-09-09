"""Two-stage FCFF discounted cash-flow model.

Modelling conventions, and why each one is there
------------------------------------------------
* **Mid-year discounting** (default). Cash flows arrive through the year, not
  in a lump on 31 December. End-year discounting understates value by roughly
  half a year of WACC — about 4% at a 9% cost of capital — which is larger
  than most of the "insights" a junior model argues about.
* **Terminal value from a steady-state reinvestment rate**, not from growing
  the last forecast year's free cash flow. Gordon-on-FCFF silently assumes the
  final explicit year's capex/D&A relationship holds to infinity; if the
  company is in an investment phase that assumption alone can move fair value
  by 30%. Steady state instead requires reinvestment = g / ROIC, so growth has
  to be paid for.
* **Working capital scales with incremental revenue**, not with the revenue
  level: net working capital is a stock proportional to sales, so its *flow*
  is proportional to the *change* in sales.
* **Terminal value share is reported**, because a DCF where 90% of the value
  sits beyond the forecast horizon is a terminal-value model wearing a DCF
  costume, and the reader is entitled to know that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantdesk.valuation.wacc import CostOfCapital

TERMINAL_METHODS = ("reinvestment", "gordon_fcff")


@dataclass(frozen=True)
class DCFResult:
    """Full output of one DCF run, including the projection for inspection."""

    ticker: str
    enterprise_value: float
    equity_value: float
    value_per_share: float
    current_price: float
    upside: float
    wacc: float
    terminal_growth: float
    terminal_value: float
    pv_terminal_value: float
    pv_explicit: float
    terminal_value_share: float
    implied_exit_ev_ebitda: float
    terminal_roic: float
    net_debt: float
    shares: float
    projection: pd.DataFrame = field(repr=False)
    cost_of_capital: CostOfCapital | None = field(default=None, repr=False)

    @property
    def recommendation(self) -> str:
        """A simple, stated band — not advice, just a consistent labelling rule."""
        if not np.isfinite(self.upside):
            return "N/A"
        if self.upside >= 0.20:
            return "BUY"
        if self.upside <= -0.20:
            return "SELL"
        return "HOLD"

    def to_row(self) -> dict[str, float | str]:
        return {
            "Price": self.current_price,
            "Fair value": self.value_per_share,
            "Upside": self.upside,
            "WACC": self.wacc,
            "Terminal g": self.terminal_growth,
            "TV share of EV": self.terminal_value_share,
            "Implied exit EV/EBITDA": self.implied_exit_ev_ebitda,
            "Signal": self.recommendation,
        }


def _project(
    *,
    revenue: float,
    revenue_growth: list[float],
    ebit_margin: list[float],
    d_and_a: float,
    capex: float,
    delta_nwc: float,
    tax_rate: float,
) -> pd.DataFrame:
    """Build the explicit forecast of unlevered free cash flow."""
    if len(revenue_growth) != len(ebit_margin):
        raise ValueError("revenue_growth and ebit_margin must have the same length")
    if revenue <= 0:
        raise ValueError("base revenue must be positive")

    da_intensity = d_and_a / revenue
    capex_intensity = capex / revenue

    # NWC intensity is calibrated off the base-year flow against the first
    # forecast year's incremental revenue, then held constant.
    base_increment = revenue * revenue_growth[0]
    nwc_intensity = delta_nwc / base_increment if base_increment else 0.0

    rows = []
    prev_revenue = revenue
    for year, (g, margin) in enumerate(zip(revenue_growth, ebit_margin, strict=True), start=1):
        rev = prev_revenue * (1.0 + g)
        ebit = rev * margin
        nopat = ebit * (1.0 - tax_rate)
        da = rev * da_intensity
        cap = rev * capex_intensity
        dnwc = (rev - prev_revenue) * nwc_intensity
        rows.append(
            {
                "year": year,
                "revenue": rev,
                "revenue_growth": g,
                "ebit": ebit,
                "ebit_margin": margin,
                "nopat": nopat,
                "d_and_a": da,
                "capex": cap,
                "delta_nwc": dnwc,
                "ebitda": ebit + da,
                "fcff": nopat + da - cap - dnwc,
            }
        )
        prev_revenue = rev

    return pd.DataFrame(rows).set_index("year")


def run_dcf(
    ticker: str,
    inputs: dict,
    *,
    market: dict,
    current_price: float,
    cost_of_capital: CostOfCapital | None = None,
    wacc_override: float | None = None,
    terminal_growth_override: float | None = None,
    terminal_method: str = "reinvestment",
    terminal_roic_spread: float = 0.02,
    mid_year: bool = True,
) -> DCFResult:
    """Value one non-financial company.

    ``inputs`` is one entry of ``config/fundamentals.yaml → dcf``.
    ``market`` supplies the risk-free rate and equity risk premium.
    """
    if terminal_method not in TERMINAL_METHODS:
        raise ValueError(f"terminal_method must be one of {TERMINAL_METHODS}")

    tax_rate = float(inputs["tax_rate"])
    shares = float(inputs["shares_diluted"])
    net_debt = float(inputs["total_debt"]) - float(inputs["cash_and_investments"])

    if cost_of_capital is None:
        from quantdesk.valuation.wacc import compute_wacc

        cost_of_capital = compute_wacc(
            market_cap=current_price * shares,
            total_debt=float(inputs["total_debt"]),
            beta=float(inputs["beta"]),
            risk_free_rate=float(market["risk_free_rate"]),
            equity_risk_premium=float(market["equity_risk_premium"]),
            cost_of_debt=float(inputs["cost_of_debt"]),
            tax_rate=tax_rate,
        )

    wacc = float(wacc_override if wacc_override is not None else cost_of_capital.wacc)
    g = float(
        terminal_growth_override
        if terminal_growth_override is not None
        else inputs["terminal_growth"]
    )
    if wacc <= g:
        raise ValueError(f"{ticker}: WACC ({wacc:.4f}) must exceed terminal growth ({g:.4f})")

    projection = _project(
        revenue=float(inputs["revenue"]),
        revenue_growth=[float(x) for x in inputs["revenue_growth"]],
        ebit_margin=[float(x) for x in inputs["ebit_margin"]],
        d_and_a=float(inputs["d_and_a"]),
        capex=float(inputs["capex"]),
        delta_nwc=float(inputs["delta_nwc"]),
        tax_rate=tax_rate,
    )

    n = len(projection)
    offset = 0.5 if mid_year else 0.0
    periods = projection.index.to_numpy(dtype=float) - offset
    discount_factors = 1.0 / (1.0 + wacc) ** periods
    projection = projection.assign(
        discount_factor=discount_factors,
        pv_fcff=projection["fcff"].to_numpy() * discount_factors,
    )
    pv_explicit = float(projection["pv_fcff"].sum())

    # --- terminal value ----------------------------------------------------
    terminal_nopat = float(projection["nopat"].iloc[-1]) * (1.0 + g)
    if terminal_method == "reinvestment":
        # Steady-state economics: growth must be bought with capital, and the
        # price of that growth is the return the franchise earns on it.
        terminal_roic = float(inputs.get("terminal_roic") or (wacc + terminal_roic_spread))
        if terminal_roic <= g:
            raise ValueError(
                f"{ticker}: terminal ROIC ({terminal_roic:.3f}) must exceed terminal growth"
            )
        reinvestment_rate = g / terminal_roic
        terminal_fcff = terminal_nopat * (1.0 - reinvestment_rate)
    else:
        terminal_roic = np.nan
        terminal_fcff = float(projection["fcff"].iloc[-1]) * (1.0 + g)

    terminal_value = terminal_fcff / (wacc - g)
    # The terminal value sits at the end of year N regardless of the mid-year
    # convention applied to the flows inside the forecast window.
    pv_terminal = terminal_value / (1.0 + wacc) ** n

    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares
    upside = value_per_share / current_price - 1.0 if current_price > 0 else np.nan

    terminal_ebitda = float(projection["ebitda"].iloc[-1]) * (1.0 + g)
    implied_exit_multiple = terminal_value / terminal_ebitda if terminal_ebitda > 0 else np.nan

    return DCFResult(
        ticker=ticker,
        enterprise_value=enterprise_value,
        equity_value=equity_value,
        value_per_share=value_per_share,
        current_price=current_price,
        upside=float(upside),
        wacc=wacc,
        terminal_growth=g,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal,
        pv_explicit=pv_explicit,
        terminal_value_share=pv_terminal / enterprise_value if enterprise_value else np.nan,
        implied_exit_ev_ebitda=float(implied_exit_multiple),
        terminal_roic=float(terminal_roic),
        net_debt=net_debt,
        shares=shares,
        projection=projection,
        cost_of_capital=cost_of_capital,
    )


def sensitivity_grid(
    ticker: str,
    inputs: dict,
    *,
    market: dict,
    current_price: float,
    base: DCFResult,
    wacc_range: np.ndarray | None = None,
    growth_range: np.ndarray | None = None,
    metric: str = "value_per_share",
    **dcf_kwargs,
) -> pd.DataFrame:
    """Classic WACC x terminal-growth football-grid.

    Rows are WACC, columns are terminal growth. Infeasible cells (g >= WACC)
    are returned as NaN rather than as a spuriously large number.
    """
    if metric not in ("value_per_share", "upside"):
        raise ValueError("metric must be 'value_per_share' or 'upside'")

    wacc_range = (
        np.round(np.linspace(base.wacc - 0.02, base.wacc + 0.02, 5), 6)
        if wacc_range is None
        else np.asarray(wacc_range, dtype=float)
    )
    growth_range = (
        np.round(np.linspace(base.terminal_growth - 0.010, base.terminal_growth + 0.010, 5), 6)
        if growth_range is None
        else np.asarray(growth_range, dtype=float)
    )

    grid = pd.DataFrame(index=wacc_range, columns=growth_range, dtype=float)
    for w in wacc_range:
        for g in growth_range:
            if w <= g + 0.005:  # keep a 50bp cushion; the ratio explodes at the boundary
                continue
            try:
                result = run_dcf(
                    ticker, inputs, market=market, current_price=current_price,
                    cost_of_capital=base.cost_of_capital, wacc_override=float(w),
                    terminal_growth_override=float(g), **dcf_kwargs,
                )
                grid.loc[w, g] = getattr(result, metric)
            except ValueError:
                continue

    grid.index.name = "WACC"
    grid.columns.name = "Terminal growth"
    return grid
