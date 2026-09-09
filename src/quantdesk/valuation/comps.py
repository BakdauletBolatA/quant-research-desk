"""Trading comparables and the football-field summary.

A DCF tells you what a business is worth; comps tell you what the market is
currently paying for businesses like it. Neither is a valuation on its own —
the honest output is a *range*, which is why every name ends up on a football
field rather than at a single price target.

Multiples are computed on an enterprise-value basis wherever the numerator is
available to all capital providers (sales, EBITDA, EBIT) and on an equity basis
only for P/E. Mixing the two — the classic ``EV / net income`` — is a capital
structure error, not a rounding difference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MULTIPLE_COLUMNS = ["EV/Sales", "EV/EBITDA", "EV/EBIT", "P/E"]


def _estimate_net_income(inputs: dict) -> float:
    """Net income proxy: EBIT less estimated interest expense, taxed."""
    interest = float(inputs["total_debt"]) * float(inputs["cost_of_debt"])
    pre_tax = float(inputs["ebit"]) - interest
    return pre_tax * (1.0 - float(inputs["tax_rate"]))


def comps_table(
    dcf_inputs: dict[str, dict],
    prices: dict[str, float],
    sectors: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Current trading multiples for every name in the DCF input sheet."""
    rows: dict[str, dict[str, float | str]] = {}

    for ticker, inputs in dcf_inputs.items():
        price = prices.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue

        shares = float(inputs["shares_diluted"])
        market_cap = price * shares
        net_debt = float(inputs["total_debt"]) - float(inputs["cash_and_investments"])
        enterprise_value = market_cap + net_debt

        revenue = float(inputs["revenue"])
        ebit = float(inputs["ebit"])
        ebitda = ebit + float(inputs["d_and_a"])
        net_income = _estimate_net_income(inputs)
        nopat = ebit * (1.0 - float(inputs["tax_rate"]))
        fcff = nopat + float(inputs["d_and_a"]) - float(inputs["capex"]) - float(inputs["delta_nwc"])

        rows[ticker] = {
            "Sector": (sectors or {}).get(ticker, ""),
            "Price": price,
            "Market cap": market_cap,
            "Net debt": net_debt,
            "EV": enterprise_value,
            "EV/Sales": enterprise_value / revenue if revenue > 0 else np.nan,
            "EV/EBITDA": enterprise_value / ebitda if ebitda > 0 else np.nan,
            "EV/EBIT": enterprise_value / ebit if ebit > 0 else np.nan,
            "P/E": market_cap / net_income if net_income > 0 else np.nan,
            "FCFF yield on EV": fcff / enterprise_value if enterprise_value > 0 else np.nan,
            "EBIT margin": ebit / revenue if revenue > 0 else np.nan,
            "Rev. growth (Y1E)": float(inputs["revenue_growth"][0]),
        }

    table = pd.DataFrame(rows).T
    if table.empty:
        return table

    numeric = [c for c in table.columns if c != "Sector"]
    table[numeric] = table[numeric].astype(float)
    return table


def peer_statistics(table: pd.DataFrame, exclude: str | None = None) -> pd.DataFrame:
    """Quartiles and median of each multiple across the peer set."""
    peers = table.drop(index=exclude, errors="ignore") if exclude else table
    stats = peers[MULTIPLE_COLUMNS].astype(float)
    return pd.DataFrame(
        {
            "P25": stats.quantile(0.25),
            "Median": stats.median(),
            "P75": stats.quantile(0.75),
        }
    )


def implied_price_from_multiple(
    inputs: dict, multiple: float, basis: str
) -> float:
    """Convert a peer multiple into an implied share price for one company."""
    shares = float(inputs["shares_diluted"])
    net_debt = float(inputs["total_debt"]) - float(inputs["cash_and_investments"])

    if basis == "EV/Sales":
        enterprise_value = multiple * float(inputs["revenue"])
    elif basis == "EV/EBITDA":
        enterprise_value = multiple * (float(inputs["ebit"]) + float(inputs["d_and_a"]))
    elif basis == "EV/EBIT":
        enterprise_value = multiple * float(inputs["ebit"])
    elif basis == "P/E":
        return multiple * _estimate_net_income(inputs) / shares
    else:
        raise ValueError(f"Unsupported basis {basis!r}")

    return (enterprise_value - net_debt) / shares


def football_field(
    ticker: str,
    inputs: dict,
    comps: pd.DataFrame,
    *,
    current_price: float,
    dcf_low: float,
    dcf_high: float,
    sensitivity_low: float,
    sensitivity_high: float,
    price_52w_low: float | None = None,
    price_52w_high: float | None = None,
) -> pd.DataFrame:
    """Value ranges per methodology, ready to plot as a football field."""
    peers = peer_statistics(comps, exclude=ticker)
    rows: list[dict[str, float | str]] = [
        {"method": "DCF — Monte Carlo (P25–P75)", "low": dcf_low, "high": dcf_high},
        {
            "method": "DCF — WACC/g sensitivity",
            "low": sensitivity_low,
            "high": sensitivity_high,
        },
    ]

    for basis in ("EV/EBITDA", "EV/EBIT", "P/E"):
        low_mult, high_mult = peers.loc[basis, "P25"], peers.loc[basis, "P75"]
        if not (np.isfinite(low_mult) and np.isfinite(high_mult)):
            continue
        low = implied_price_from_multiple(inputs, float(low_mult), basis)
        high = implied_price_from_multiple(inputs, float(high_mult), basis)
        rows.append(
            {"method": f"Peers — {basis} (P25–P75)", "low": min(low, high), "high": max(low, high)}
        )

    if price_52w_low is not None and price_52w_high is not None:
        rows.append({"method": "52-week trading range", "low": price_52w_low,
                     "high": price_52w_high})

    field = pd.DataFrame(rows).set_index("method")
    field["mid"] = (field["low"] + field["high"]) / 2.0
    field["implied upside (mid)"] = field["mid"] / current_price - 1.0
    return field
