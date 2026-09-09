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
    "compute_wacc",
    "DCFResult",
    "run_dcf",
    "sensitivity_grid",
    "ExcessReturnResult",
    "run_excess_return_model",
    "MonteCarloResult",
    "run_monte_carlo",
    "ReverseDCFResult",
    "reverse_dcf",
    "comps_table",
    "football_field",
    "peer_statistics",
    "implied_price_from_multiple",
]
