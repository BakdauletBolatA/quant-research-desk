"""Excess-return (residual income) valuation for banks.

Why banks are not valued with FCFF
----------------------------------
Free cash flow to the firm is defined as cash available to *all* capital
providers before financing. For a bank, financing **is** the operation: debt is
raw material, and "working capital" is the loan book. Capex and net working
capital have no clean meaning, so the FCFF bridge breaks. Valuing JPM with a
FCFF DCF is the fastest way to fail a technical interview.

The excess-return form is equivalent to a dividend discount model but far more
robust, because it anchors on book equity — which banks actually report and
mark — and only discounts the *spread* the franchise earns over its cost of
equity:

    V0 = BV0 + Σ (ROE_t − Ke) · BV_(t−1) / (1 + Ke)^t + PV(terminal spread)

If ROE = Ke forever, the bank is worth exactly book. Everything above book has
to be earned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantdesk.valuation.wacc import cost_of_equity_capm


@dataclass(frozen=True)
class ExcessReturnResult:
    ticker: str
    equity_value: float
    value_per_share: float
    current_price: float
    upside: float
    cost_of_equity: float
    book_value_per_share: float
    implied_p_b: float
    current_p_b: float
    terminal_share: float
    projection: pd.DataFrame = field(repr=False)

    @property
    def recommendation(self) -> str:
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
            "Cost of equity": self.cost_of_equity,
            "BVPS": self.book_value_per_share,
            "Implied P/B": self.implied_p_b,
            "Current P/B": self.current_p_b,
            "TV share": self.terminal_share,
            "Signal": self.recommendation,
        }


def run_excess_return_model(
    ticker: str,
    inputs: dict,
    *,
    market: dict,
    current_price: float,
    cost_of_equity_override: float | None = None,
) -> ExcessReturnResult:
    """Value a bank from its book equity and the spread of ROE over Ke."""
    book_value = float(inputs["book_value_equity"])
    shares = float(inputs["shares_diluted"])
    roe_path = [float(x) for x in inputs["roe_path"]]
    payout = float(inputs["payout_ratio"])
    g = float(inputs["terminal_growth"])
    terminal_roe = float(inputs["terminal_roe"])

    ke = float(
        cost_of_equity_override
        if cost_of_equity_override is not None
        else cost_of_equity_capm(
            float(market["risk_free_rate"]),
            float(inputs["beta"]),
            float(market["equity_risk_premium"]),
        )
    )
    if ke <= g:
        raise ValueError(f"{ticker}: cost of equity ({ke:.4f}) must exceed growth ({g:.4f})")

    rows = []
    opening_book = book_value
    pv_excess = 0.0
    for year, roe in enumerate(roe_path, start=1):
        net_income = opening_book * roe
        excess = (roe - ke) * opening_book
        discount = 1.0 / (1.0 + ke) ** year
        pv_excess += excess * discount
        retained = net_income * (1.0 - payout)
        rows.append(
            {
                "year": year,
                "opening_book": opening_book,
                "roe": roe,
                "net_income": net_income,
                "excess_return": excess,
                "pv_excess_return": excess * discount,
                "dividend": net_income * payout,
                "closing_book": opening_book + retained,
            }
        )
        opening_book += retained

    projection = pd.DataFrame(rows).set_index("year")

    terminal_excess = (terminal_roe - ke) * opening_book
    terminal_value = terminal_excess / (ke - g)
    pv_terminal = terminal_value / (1.0 + ke) ** len(roe_path)

    equity_value = book_value + pv_excess + pv_terminal
    value_per_share = equity_value / shares
    bvps = book_value / shares

    return ExcessReturnResult(
        ticker=ticker,
        equity_value=equity_value,
        value_per_share=value_per_share,
        current_price=current_price,
        upside=float(value_per_share / current_price - 1.0) if current_price > 0 else np.nan,
        cost_of_equity=ke,
        book_value_per_share=bvps,
        implied_p_b=value_per_share / bvps if bvps else np.nan,
        current_p_b=current_price / bvps if bvps else np.nan,
        terminal_share=pv_terminal / equity_value if equity_value else np.nan,
        projection=projection,
    )
