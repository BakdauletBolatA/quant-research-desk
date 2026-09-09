"""Intrinsic valuation: FCFF/DCF, bank excess-return, Monte Carlo and comps."""

from __future__ import annotations

from quantdesk.valuation.comps import (
    comps_table,
    football_field,
    implied_price_from_multiple,
    peer_statistics,
)
from quantdesk.valuation.dcf import DCFResult, run_dcf, sensitivity_grid
from quantdesk.valuation.excess_return import ExcessReturnResult, run_excess_return_model
from quantdesk.valuation.monte_carlo import MonteCarloResult, run_monte_carlo
from quantdesk.valuation.reverse_dcf import ReverseDCFResult, reverse_dcf
from quantdesk.valuation.wacc import CostOfCapital, compute_wacc

__all__ = [
    "CostOfCapital",
    "DCFResult",
    "ExcessReturnResult",
    "MonteCarloResult",
    "ReverseDCFResult",
    "comps_table",
    "compute_wacc",
    "football_field",
    "implied_price_from_multiple",
    "peer_statistics",
    "reverse_dcf",
    "run_dcf",
    "run_excess_return_model",
    "run_monte_carlo",
    "sensitivity_grid",
]
