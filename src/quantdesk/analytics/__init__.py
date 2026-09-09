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
    calmar_ratio,
    cumulative_return,
    drawdown_table,
    annualised_volatility,
    deflated_sharpe_ratio,
    drawdown_series,
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
    "annualised_return",
    "cumulative_return",
    "calmar_ratio",
    "drawdown_table",
    "annualised_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "drawdown_series",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "performance_summary",
    "value_at_risk",
    "expected_shortfall",
    "var_backtest",
    "var_backtest_table",
    "VarBacktest",
    "ewma_volatility",
    "rolling_volatility",
    "rolling_var_forecast",
    "rolling_beta",
    "risk_summary",
    "FactorRegression",
    "factor_regression",
    "factor_table",
    "return_attribution",
]
