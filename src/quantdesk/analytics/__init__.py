"""Performance, risk and factor analytics."""

from __future__ import annotations

from quantdesk.analytics.factor_model import (
    FactorRegression,
    factor_regression,
    factor_table,
    return_attribution,
)
from quantdesk.analytics.performance import (
    annualised_return,
    annualised_volatility,
    calmar_ratio,
    cumulative_return,
    deflated_sharpe_ratio,
    drawdown_series,
    drawdown_table,
    max_drawdown,
    performance_summary,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
)
from quantdesk.analytics.risk import (
    VarBacktest,
    ewma_volatility,
    expected_shortfall,
    risk_summary,
    rolling_beta,
    rolling_var_forecast,
    rolling_volatility,
    value_at_risk,
    var_backtest,
    var_backtest_table,
)

__all__ = [
    "FactorRegression",
    "VarBacktest",
    "annualised_return",
    "annualised_volatility",
    "calmar_ratio",
    "cumulative_return",
    "deflated_sharpe_ratio",
    "drawdown_series",
    "drawdown_table",
    "ewma_volatility",
    "expected_shortfall",
    "factor_regression",
    "factor_table",
    "max_drawdown",
    "performance_summary",
    "probabilistic_sharpe_ratio",
    "return_attribution",
    "risk_summary",
    "rolling_beta",
    "rolling_var_forecast",
    "rolling_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "value_at_risk",
    "var_backtest",
    "var_backtest_table",
]
