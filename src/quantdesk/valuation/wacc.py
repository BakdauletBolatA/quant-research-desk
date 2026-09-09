"""Cost of capital.

Deliberate choices worth defending in an interview:

* **Market-value weights, not book.** Using book equity in the WACC weights is
  the most common error in a junior model; for a company trading at 8x book it
  overstates the debt weight by an order of magnitude.
* **Book debt as a proxy for market debt.** Acceptable for investment-grade
  issuers trading near par, and stated as an assumption rather than hidden.
* **Levered beta taken as given from the input sheet.** The Hamada re-levering
  helper is provided for peer-beta work, but silently re-levering a beta that
  is already levered is another classic own-goal, so it is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostOfCapital:
    """Decomposed WACC with every input retained for the audit trail."""

    cost_of_equity: float
    after_tax_cost_of_debt: float
    equity_weight: float
    debt_weight: float
    wacc: float
    risk_free_rate: float
    equity_risk_premium: float
    beta: float
    market_cap: float
    total_debt: float

    def to_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def cost_of_equity_capm(risk_free_rate: float, beta: float, equity_risk_premium: float) -> float:
    """Ke = rf + beta * ERP."""
    return float(risk_free_rate + beta * equity_risk_premium)


def unlever_beta(levered_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    """Hamada: asset beta implied by an observed equity beta."""
    return float(levered_beta / (1.0 + (1.0 - tax_rate) * debt_to_equity))


def relever_beta(asset_beta: float, debt_to_equity: float, tax_rate: float) -> float:
    """Hamada: equity beta implied by a target capital structure."""
    return float(asset_beta * (1.0 + (1.0 - tax_rate) * debt_to_equity))


def compute_wacc(
    *,
    market_cap: float,
    total_debt: float,
    beta: float,
    risk_free_rate: float,
    equity_risk_premium: float,
    cost_of_debt: float,
    tax_rate: float,
) -> CostOfCapital:
    """Market-value-weighted WACC."""
    if market_cap <= 0:
        raise ValueError("market_cap must be positive")
    if total_debt < 0:
        raise ValueError("total_debt cannot be negative")

    capital = market_cap + total_debt
    equity_weight = market_cap / capital
    debt_weight = total_debt / capital

    ke = cost_of_equity_capm(risk_free_rate, beta, equity_risk_premium)
    kd_after_tax = cost_of_debt * (1.0 - tax_rate)
    wacc = equity_weight * ke + debt_weight * kd_after_tax

    return CostOfCapital(
        cost_of_equity=ke,
        after_tax_cost_of_debt=kd_after_tax,
        equity_weight=equity_weight,
        debt_weight=debt_weight,
        wacc=float(wacc),
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        beta=beta,
        market_cap=market_cap,
        total_debt=total_debt,
    )
