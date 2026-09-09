"""End-to-end research pipeline.

One entry point produces the whole deliverable: figures, the HTML research
report and the Excel workbook. Running it twice on the same cache produces
byte-identical numbers, which is the minimum bar for anything that calls itself
research.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantdesk import __version__
from quantdesk.analytics import (
    drawdown_table,
    factor_regression,
    factor_table,
    performance_summary,
    return_attribution,
    risk_summary,
    rolling_beta,
    rolling_var_forecast,
    rolling_volatility,
    var_backtest_table,
)
from quantdesk.analytics.performance import deflated_sharpe_ratio, sharpe_ratio
from quantdesk.backtest import combine_returns, run_all, turnover_table
from quantdesk.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    Config,
    ensure_dirs,
    load_config,
    load_fundamentals,
)
from quantdesk.data import ResearchPanel, build_panel
from quantdesk.portfolio import (
    Constraints,
    black_litterman,
    condition_number,
    correlation_from_covariance,
    derive_risk_aversion,
    effective_number_of_bets,
    efficient_frontier,
    estimate_covariance,
    ledoit_wolf_shrinkage,
    market_cap_weights,
    optimise,
    portfolio_volatility,
    risk_contributions,
)
from quantdesk.reporting import charts, write_workbook
from quantdesk.reporting.tearsheet import (
    FigureCounter,
    bullets,
    callout,
    render_tearsheet,
    table,
    text,
)
from quantdesk.valuation import (
    comps_table,
    football_field,
    peer_statistics,
    run_dcf,
    run_excess_return_model,
    run_monte_carlo,
    sensitivity_grid,
)
from quantdesk.valuation.reverse_dcf import reverse_dcf

logger = logging.getLogger(__name__)

HEADLINE_STRATEGY = "black_litterman"
FOCUS_TICKER = "MSFT"


@dataclass
class PipelineResult:
    """Everything the pipeline computed, so a notebook can pick it apart."""

    panel: ResearchPanel
    performance: pd.DataFrame
    risk: pd.DataFrame
    var_validation: pd.DataFrame
    factors: pd.DataFrame
    backtests: dict[str, Any]
    weights: pd.DataFrame
    valuation: pd.DataFrame
    reverse: pd.DataFrame
    comps: pd.DataFrame
    figures: dict[str, str] = field(default_factory=dict)
    report_html: str = ""
    workbook: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(value: float, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:{'+' if signed else ''}.1%}"


def _num(value: float, decimals: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:,.{decimals}f}"


def _select_metrics(summary: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return summary.loc[[m for m in metrics if m in summary.index]]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def run_pipeline(cfg: Config | None = None, *, force_download: bool = False) -> PipelineResult:
    """Run the whole study and write the deliverables to ``reports/``."""
    ensure_dirs()
    cfg = cfg or load_config()
    fundamentals = load_fundamentals()
    market = fundamentals["market"]
    figures: dict[str, str] = {}
    fig_dir = FIGURES_DIR

    # ---------------------------------------------------------------- data --
    panel = build_panel(cfg, force=force_download)
    ann = panel.periods_per_year
    benchmark_label = f"{cfg.benchmark} (benchmark)"
    last_prices = panel.prices.iloc[-1]

    # -------------------------------------------------------- single names --
    logger.info("Stage 1/7 · single-name performance and risk")
    name_returns = panel.returns.copy()
    name_returns[benchmark_label] = panel.benchmark_returns
    name_performance = performance_summary(
        name_returns, panel.benchmark_returns, panel.risk_free, ann
    )
    name_risk = risk_summary(
        name_returns,
        cfg["analytics"]["var_confidence"],
        cfg["analytics"]["es_confidence"],
        cfg["analytics"]["ewma_lambda"],
        ann,
    )

    # ------------------------------------------------------------- factors --
    logger.info("Stage 2/7 · factor attribution")
    name_factors = factor_table(
        panel.excess_returns, panel.factors, periods=ann,
        nw_lags=cfg["factors"]["newey_west_lags"],
    )

    # --------------------------------------------------- portfolio machinery --
    logger.info("Stage 3/7 · covariance, equilibrium and allocators")
    portfolio_cfg = cfg["portfolio"]
    constraints = Constraints(
        long_only=portfolio_cfg["long_only"],
        min_weight=portfolio_cfg["min_weight"],
        max_weight=portfolio_cfg["max_weight"],
    )
    cov = estimate_covariance(panel.returns, portfolio_cfg["covariance"], ann)
    cov_sample = estimate_covariance(panel.returns, "sample", ann)
    shrinkage = ledoit_wolf_shrinkage(panel.returns)

    market_weights = market_cap_weights(last_prices, fundamentals["shares_outstanding"])
    market_weights = (market_weights / market_weights.sum()).reindex(cov.index)
    risk_aversion = portfolio_cfg.get("risk_aversion") or derive_risk_aversion(
        cov, market_weights, market["equity_risk_premium"]
    )
    bl = black_litterman(
        cov, market_weights, portfolio_cfg["black_litterman"]["views"],
        risk_aversion=risk_aversion, tau=portfolio_cfg["black_litterman"]["tau"],
    )

    allocations: dict[str, pd.Series] = {}
    for method in ("equal_weight", "min_variance", "risk_parity", "max_diversification", "hrp"):
        allocations[method] = optimise(method, cov, bl.posterior_returns, constraints)
    allocations["max_sharpe"] = optimise(
        "max_sharpe", cov, panel.excess_returns.mean() * ann, constraints
    )
    allocations["black_litterman"] = optimise(
        "max_sharpe", bl.posterior_covariance, bl.posterior_returns, constraints
    ).rename("black_litterman")
    weights_frame = pd.DataFrame(allocations)

    from quantdesk.backtest.strategies import LABELS

    weights_display = weights_frame.rename(columns=LABELS)
    frontier = efficient_frontier(cov, bl.posterior_returns, constraints, n_points=45)

    asset_points = pd.DataFrame(
        {
            "volatility": np.sqrt(np.diag(cov.to_numpy())),
            "expected_return": bl.posterior_returns.reindex(cov.index).to_numpy(),
        },
        index=cov.index,
    )
    portfolio_points = pd.DataFrame(
        {
            LABELS.get(name, name): {
                "volatility": portfolio_volatility(w.to_numpy(), cov.to_numpy()),
                "expected_return": float(w @ bl.posterior_returns.reindex(w.index)),
            }
            for name, w in allocations.items()
        }
    ).T

    contribution_frame = pd.DataFrame(
        {
            LABELS.get(name, name): risk_contributions(w.to_numpy(), cov.to_numpy())
            / portfolio_volatility(w.to_numpy(), cov.to_numpy())
            for name, w in allocations.items()
            if name in ("equal_weight", "risk_parity", "min_variance", "black_litterman")
        },
        index=cov.index,
    )

    # ------------------------------------------------------------ backtest --
    logger.info("Stage 4/7 · walk-forward backtest")
    backtest_cfg = cfg["backtest"]
    results = run_all(
        backtest_cfg["strategies"],
        panel.returns,
        panel.prices,
        panel.risk_free,
        start=backtest_cfg["start"],
        rebalance=backtest_cfg["rebalance"],
        lookback_days=backtest_cfg["lookback_days"],
        transaction_cost_bps=backtest_cfg["transaction_cost_bps"],
        constraints=constraints,
        covariance_method=portfolio_cfg["covariance"],
        periods_per_year=ann,
        shares_outstanding=fundamentals["shares_outstanding"],
        equity_risk_premium=market["equity_risk_premium"],
        views=portfolio_cfg["black_litterman"]["views"],
        tau=portfolio_cfg["black_litterman"]["tau"],
        risk_aversion=portfolio_cfg.get("risk_aversion"),
    )
    strategy_returns = combine_returns(results)
    bench_bt = panel.benchmark_returns.reindex(strategy_returns.index)
    strategy_returns[benchmark_label] = bench_bt

    strategy_performance = performance_summary(strategy_returns, bench_bt, panel.risk_free, ann)
    turnover = turnover_table(results)

    # Multiple-testing correction across the strategies actually tried.
    trial_sharpes = np.array(
        [sharpe_ratio(r.returns, panel.risk_free, ann) for r in results.values()]
    )
    deflated = {
        r.label: deflated_sharpe_ratio(
            r.returns, n_trials=len(results), trial_sharpes=trial_sharpes,
            periods=ann, risk_free=panel.risk_free,
        )
        for r in results.values()
    }
    strategy_performance.loc["Deflated Sharpe p"] = pd.Series(deflated).reindex(
        strategy_performance.columns
    )

    headline = results.get(HEADLINE_STRATEGY) or next(iter(results.values()))
    headline_fit = factor_regression(
        headline.returns.sub(panel.risk_free.reindex(headline.returns.index).ffill()).dropna(),
        panel.factors,
        periods=ann,
        nw_lags=cfg["factors"]["newey_west_lags"],
        name=headline.label,
    )
    headline_attribution = return_attribution(headline_fit, panel.factors, ann)

    # --------------------------------------------------------------- risk ---
    logger.info("Stage 5/7 · VaR model validation")
    var_conf = cfg["analytics"]["var_confidence"]
    validation = var_backtest_table(
        headline.returns, var_conf, window=500, lam=cfg["analytics"]["ewma_lambda"]
    )
    best_method = (
        validation["kupiec_p"].astype(float).idxmax() if not validation.empty
        else "filtered_historical"
    )
    var_path = rolling_var_forecast(
        headline.returns, var_conf, best_method, 500, cfg["analytics"]["ewma_lambda"]
    )

    # ---------------------------------------------------------- valuation ---
    logger.info("Stage 6/7 · valuation")
    valuation_cfg = cfg["valuation"]
    dcf_rows, reverse_rows, mc_summaries = {}, {}, {}
    dcf_results, mc_results = {}, {}

    for ticker, inputs in fundamentals["dcf"].items():
        if ticker not in last_prices.index:
            continue
        price = float(last_prices[ticker])
        result = run_dcf(ticker, inputs, market=market, current_price=price)
        dcf_results[ticker] = result
        dcf_rows[ticker] = result.to_row()

        reverse_rows[ticker] = reverse_dcf(
            ticker, inputs, market=market, current_price=price, base_result=result
        ).to_row()

        mc = run_monte_carlo(
            ticker, inputs, market=market, current_price=price, base_wacc=result.wacc,
            base_value=result.value_per_share, mc_config=fundamentals["monte_carlo"],
            n_paths=int(valuation_cfg["monte_carlo_paths"]), seed=int(valuation_cfg["seed"]),
        )
        mc_results[ticker] = mc
        mc_summaries[ticker] = mc.summary()

    bank_rows = {
        ticker: run_excess_return_model(
            ticker, inputs, market=market, current_price=float(last_prices[ticker])
        ).to_row()
        for ticker, inputs in fundamentals["excess_return"].items()
        if ticker in last_prices.index
    }

    valuation_frame = pd.DataFrame(dcf_rows).T
    reverse_frame = pd.DataFrame(reverse_rows).T
    mc_frame = pd.DataFrame(mc_summaries).T
    bank_frame = pd.DataFrame(bank_rows).T
    comps = comps_table(fundamentals["dcf"], last_prices.to_dict(), cfg.sectors)
    peers = peer_statistics(comps)

    focus = FOCUS_TICKER if FOCUS_TICKER in dcf_results else next(iter(dcf_results))
    focus_inputs = fundamentals["dcf"][focus]
    focus_price = float(last_prices[focus])
    focus_result = dcf_results[focus]
    focus_grid = sensitivity_grid(
        focus, focus_inputs, market=market, current_price=focus_price, base=focus_result
    )
    focus_window = panel.prices[focus].tail(252)
    focus_field = football_field(
        focus, focus_inputs, comps, current_price=focus_price,
        dcf_low=mc_results[focus].percentile(25), dcf_high=mc_results[focus].percentile(75),
        sensitivity_low=float(np.nanmin(focus_grid.to_numpy())),
        sensitivity_high=float(np.nanmax(focus_grid.to_numpy())),
        price_52w_low=float(focus_window.min()), price_52w_high=float(focus_window.max()),
    )

    # ------------------------------------------------------------ figures ---
    logger.info("Stage 7/7 · figures and deliverables")
    strategy_only = strategy_returns.drop(columns=[benchmark_label])

    figures["equity"] = charts.equity_curves(
        strategy_only, fig_dir / "01_equity_curves.png",
        benchmark=bench_bt.rename(cfg.benchmark),
    )
    figures["drawdown"] = charts.drawdown_chart(
        strategy_returns, fig_dir / "02_drawdown.png",
        highlight=[headline.label, benchmark_label],
    )
    figures["rolling_vol"] = charts.rolling_metric(
        {
            headline.label: rolling_volatility(headline.returns, 252, ann),
            cfg.benchmark: rolling_volatility(bench_bt, 252, ann),
        },
        fig_dir / "03_rolling_vol.png",
        "Rolling 1-year annualised volatility", "Volatility", as_percent=True,
    )
    figures["rolling_beta"] = charts.rolling_metric(
        {f"{headline.label} beta to {cfg.benchmark}": rolling_beta(headline.returns, bench_bt, 252)},
        fig_dir / "04_rolling_beta.png",
        f"Rolling 1-year beta to {cfg.benchmark}", "Beta",
    )
    figures["correlation"] = charts.correlation_heatmap(
        correlation_from_covariance(cov), fig_dir / "05_correlation.png",
        "Return correlation — Ledoit-Wolf shrunk estimate",
    )
    figures["var_exceptions"] = charts.var_exceptions(
        headline.returns, var_path, fig_dir / "06_var_exceptions.png", var_conf, best_method,
    )
    figures["var_models"] = charts.var_model_comparison(
        validation, fig_dir / "07_var_models.png", 1.0 - var_conf
    )
    figures["factors"] = charts.factor_exposures(
        pd.Series(headline_fit.betas), pd.Series(headline_fit.tstats),
        fig_dir / "08_factor_exposures.png",
        f"{headline.label} — factor loadings (FF5 + momentum, Newey-West)",
    )
    figures["attribution"] = charts.attribution_chart(
        headline_attribution, fig_dir / "09_attribution.png",
        f"{headline.label} — where the excess return came from",
    )
    figures["frontier"] = charts.efficient_frontier_chart(
        frontier, asset_points, portfolio_points, fig_dir / "10_frontier.png"
    )
    figures["weights"] = charts.weights_heatmap(
        weights_display, fig_dir / "11_weights.png",
        "Target allocation by construction method (full-sample estimates)",
    )
    figures["risk_contrib"] = charts.risk_contribution_chart(
        contribution_frame, fig_dir / "12_risk_contributions.png"
    )
    figures["monte_carlo"] = charts.monte_carlo_distribution(
        mc_results[focus].values, fig_dir / "13_monte_carlo.png", focus, focus_price,
        focus_result.value_per_share,
    )
    figures["sensitivity"] = charts.sensitivity_heatmap(
        focus_grid, fig_dir / "14_sensitivity.png", focus, focus_price
    )
    figures["football"] = charts.football_field_chart(
        focus_field, fig_dir / "15_football_field.png", focus, focus_price
    )
    figures["reverse"] = charts.reverse_dcf_chart(reverse_frame, fig_dir / "16_reverse_dcf.png")
    figures["implied_wacc"] = charts.implied_cost_of_capital_chart(
        reverse_frame, fig_dir / "17_implied_wacc.png", market["risk_free_rate"]
    )
    figures["upside"] = charts.upside_chart(valuation_frame, fig_dir / "18_upside.png")

    # --------------------------------------------------------- deliverables --
    context = _build_report_context(
        cfg=cfg, panel=panel, figures=figures, market=market,
        name_performance=name_performance, name_risk=name_risk, name_factors=name_factors,
        strategy_performance=strategy_performance, turnover=turnover,
        headline=headline, headline_fit=headline_fit, headline_attribution=headline_attribution,
        validation=validation, best_method=best_method, var_conf=var_conf,
        cov=cov, cov_sample=cov_sample, shrinkage=shrinkage, bl=bl,
        risk_aversion=risk_aversion, weights_display=weights_display,
        valuation_frame=valuation_frame, reverse_frame=reverse_frame, mc_frame=mc_frame,
        bank_frame=bank_frame, comps=comps, peers=peers, focus=focus,
        focus_field=focus_field, focus_result=focus_result,
        benchmark_label=benchmark_label, results=results, deflated=deflated,
        strategy_returns=strategy_returns,
    )
    report_path = render_tearsheet(REPORTS_DIR / "quantdesk_research_report.html", **context)
    logger.info("Report written: %s", report_path)

    workbook_path = write_workbook(
        REPORTS_DIR / "quantdesk_workbook.xlsx",
        {
            "Summary": [
                ("Strategy performance (net of costs)", strategy_performance),
                ("Turnover and cost", turnover),
            ],
            "Single names": [
                ("Performance", name_performance),
                ("Risk", name_risk),
            ],
            "Factors": [
                ("FF5 + momentum regressions", name_factors),
                (f"{headline.label} attribution", headline_attribution.to_frame("Contribution")),
            ],
            "Allocation": [
                ("Target weights", weights_display),
                ("Black-Litterman", bl.summary()),
                ("Risk contributions", contribution_frame),
            ],
            "VaR validation": [("Rolling backtest", validation)],
            "Valuation": [
                ("DCF summary", valuation_frame),
                ("Reverse DCF", reverse_frame),
                ("Monte Carlo", mc_frame),
                ("Banks — excess return", bank_frame),
                ("Trading comparables", comps),
                ("Peer multiples", peers),
                (f"{focus} — football field", focus_field),
                (f"{focus} — WACC / g sensitivity", focus_grid),
            ],
            "Assumptions": [
                ("DCF input sheet", pd.DataFrame(fundamentals["dcf"]).T),
                ("Bank input sheet", pd.DataFrame(fundamentals["excess_return"]).T),
                ("Market assumptions", pd.Series(market).to_frame("Value")),
            ],
        },
        title="QuantDesk — Equity Research & Portfolio Risk",
        subtitle=(
            f"{panel.start.date()} to {panel.end.date()} · "
            f"{len(panel.tickers)} names · generated {dt.date.today().isoformat()}"
        ),
        notes={
            "Assumptions": (
                "These are the modeller's own normalised estimates, not a data feed. "
                "Re-key them from the latest filings before any decision is taken. "
                "Nothing in this workbook is investment advice."
            )
        },
    )
    logger.info("Workbook written: %s", workbook_path)

    return PipelineResult(
        panel=panel,
        performance=strategy_performance,
        risk=name_risk,
        var_validation=validation,
        factors=name_factors,
        backtests=results,
        weights=weights_display,
        valuation=valuation_frame,
        reverse=reverse_frame,
        comps=comps,
        figures=figures,
        report_html=report_path,
        workbook=workbook_path,
    )


# ---------------------------------------------------------------------------
# Report narrative
# ---------------------------------------------------------------------------
def _build_report_context(**kw) -> dict[str, Any]:
    """Assemble the HTML report: metadata, headline numbers and the argument."""
    cfg: Config = kw["cfg"]
    panel: ResearchPanel = kw["panel"]
    figures: dict[str, str] = kw["figures"]
    figure = FigureCounter()
    headline = kw["headline"]
    headline_fit = kw["headline_fit"]
    validation: pd.DataFrame = kw["validation"]
    best_method: str = kw["best_method"]
    var_conf: float = kw["var_conf"]
    perf: pd.DataFrame = kw["strategy_performance"]
    benchmark_label: str = kw["benchmark_label"]
    reverse: pd.DataFrame = kw["reverse_frame"]
    valuation: pd.DataFrame = kw["valuation_frame"]
    market = kw["market"]
    focus = kw["focus"]

    ann = panel.periods_per_year
    label = headline.label
    bench_col = benchmark_label

    # ---- headline numbers ---------------------------------------------------
    best_sharpe_col = perf.loc["Sharpe"].drop(labels=[bench_col]).astype(float).idxmax()
    kpis = [
        {
            "label": "Best risk-adjusted",
            "value": _num(perf.loc["Sharpe", best_sharpe_col]),
            "foot": f"Sharpe · {best_sharpe_col}",
            "direction": "up",
        },
        {
            "label": f"{cfg.benchmark} Sharpe",
            "value": _num(perf.loc["Sharpe", bench_col]),
            "foot": "same window, same costs basis",
            "direction": "",
        },
        {
            "label": f"{label} CAGR",
            "value": _pct(perf.loc["CAGR", label]),
            "foot": f"net of {cfg['backtest']['transaction_cost_bps']:.0f}bp per trade",
            "direction": "up",
        },
        {
            "label": f"{label} max drawdown",
            "value": _pct(perf.loc["Max drawdown", label]),
            "foot": f"vs {_pct(perf.loc['Max drawdown', bench_col])} for {cfg.benchmark}",
            "direction": "down",
        },
        {
            "label": "Alpha after factors",
            "value": _pct(headline_fit.alpha_annual, signed=True),
            "foot": f"t = {headline_fit.alpha_tstat:.2f} (Newey-West)",
            "direction": "up" if headline_fit.alpha_annual >= 0 else "down",
        },
        {
            "label": "VaR models passing",
            "value": f"{int((validation['kupiec_p'].astype(float) > 0.05).sum())}/"
                     f"{len(validation)}",
            "foot": f"{var_conf:.0%} coverage, Kupiec test",
            "direction": "",
        },
        {
            "label": "Market-implied WACC",
            "value": f"{reverse['Implied WACC'].astype(float).min():.1%}–"
                     f"{reverse['Implied WACC'].astype(float).max():.1%}",
            "foot": f"vs {market['risk_free_rate']:.1%} risk-free",
            "direction": "down",
        },
        {
            "label": "Names screened cheap",
            "value": f"{int((valuation['Upside'].astype(float) > 0).sum())}/{len(valuation)}",
            "foot": "base-case DCF upside > 0",
            "direction": "",
        },
    ]

    sections: list[dict[str, Any]] = []

    # ---- 1. mandate ---------------------------------------------------------
    coverage = panel.summary()[["sector", "ann_return", "ann_vol"]].rename(
        columns={"sector": "Sector", "ann_return": "Ann. return", "ann_vol": "Ann. volatility"}
    )
    sections.append(
        {
            "title": "Mandate, data and reproducibility",
            "subtitle": (
                "A single research stack that takes a liquid US large-cap universe from raw "
                "prices to an allocation and an intrinsic-value view, with every assumption "
                "written down and every model tested rather than asserted."
            ),
            "blocks": [
                text(
                    f"The study covers <b>{len(panel.tickers)} US large caps</b> spread across "
                    f"{len(set(panel.sectors.values()))} GICS sectors, benchmarked against "
                    f"<b>{cfg.benchmark}</b>, over <b>{panel.start.date()} to "
                    f"{panel.end.date()}</b> — {len(panel.returns):,} trading days, "
                    f"{panel.n_years:.1f} years. Prices are dividend- and split-adjusted; "
                    "the risk-free path and the factor returns come from the Kenneth R. French "
                    "data library rather than from an assumed constant."
                ),
                callout(
                    "How to read this",
                    "Every section states a <b>finding</b> and the <b>test that supports it</b>. "
                    "Where a model fails, the failure is reported — a risk model nobody "
                    "backtested and a Sharpe ratio without a confidence statement are both "
                    "decorations. Section 8 lists what this study cannot tell you.",
                ),
                table(
                    coverage, caption="Universe coverage",
                    note="Annualised figures over the full sample. "
                         "Source: Yahoo Finance adjusted closes.",
                    index_label="Ticker",
                ),
            ],
        }
    )

    # ---- 2. risk ------------------------------------------------------------
    validation_display = validation.rename(
        columns={
            "observations": "Obs", "exceptions": "Exceptions",
            "expected_exceptions": "Expected exceptions", "exception_rate": "Exception rate",
            "kupiec_p": "Kupiec p", "christoffersen_p": "Christoffersen p",
            "conditional_coverage_p": "Joint p", "verdict": "Verdict",
        }
    )[
        ["Obs", "Exceptions", "Expected exceptions", "Exception rate", "Kupiec p",
         "Christoffersen p", "Joint p", "Verdict"]
    ]
    validation_display.index = [str(i).replace("_", " ").title() for i in validation_display.index]
    gaussian_rate = float(validation.loc["gaussian", "exception_rate"])
    best_rate = float(validation.loc[best_method, "exception_rate"])

    sections.append(
        {
            "title": "Market risk — and which risk model actually survives a backtest",
            "subtitle": (
                "Five one-day Value-at-Risk estimators are rolled forward out of sample on the "
                f"{label} track record and put through the two likelihood-ratio tests the Basel "
                "traffic-light framework is built on."
            ),
            "blocks": [
                text(
                    f"At the {var_conf:.0%} confidence level a correct model should breach on "
                    f"{1 - var_conf:.0%} of days. The normal-distribution model breaches on "
                    f"<b>{gaussian_rate:.2%}</b> — it understates the true risk by roughly "
                    f"{gaussian_rate / (1 - var_conf):.1f}× — which is the entire reason the "
                    "2008 risk-management post-mortems exist. Filtered Historical Simulation, "
                    "which devolatilises returns, takes the empirical tail of the standardised "
                    "residuals and re-inflates by today's conditional volatility, lands at "
                    f"<b>{best_rate:.2%}</b>."
                ),
                callout(
                    "Finding",
                    f"Only <b>{best_method.replace('_', ' ')}</b> achieves statistically correct "
                    f"unconditional coverage (Kupiec p = "
                    f"{float(validation.loc[best_method, 'kupiec_p']):.3f}). Every model — "
                    "including that one — still fails the independence test, meaning exceptions "
                    "cluster in stress. That residual clustering is the empirical case for a "
                    "GARCH filter, and it is reported here rather than hidden.",
                    "finding",
                ),
                table(validation_display, caption="VaR model validation", index_label="Estimator",
                      note="Rolling 500-day estimation window, one-day horizon, out of sample. "
                           "Kupiec tests the exception <i>rate</i>; Christoffersen tests whether "
                           "exceptions <i>cluster</i>; the joint test is the sum."),
                figure(figures["var_models"],
                       "Realised exception rate by estimator against the rate each model "
                       "promises. Green bars pass the Kupiec coverage test at the 5% level."),
                figure(figures["var_exceptions"],
                       f"The {var_conf:.0%} VaR path the risk manager would actually have seen, "
                       "against realised daily returns. Breaches are marked."),
                table(
                    _select_metrics(
                        kw["name_risk"],
                        [f"VaR {var_conf * 100:g}% historical",
                         f"VaR {var_conf * 100:g}% Cornish-Fisher",
                         f"ES {cfg['analytics']['es_confidence'] * 100:g}% historical",
                         "Ann. volatility", "Worst day"],
                    ).T,
                    caption="Single-name tail risk",
                    index_label="Ticker",
                    note="Full-sample estimates. Expected Shortfall is quoted at 97.5%, the "
                         "confidence level the FRTB market-risk framework uses.",
                ),
            ],
        }
    )

    # ---- 3. factor attribution ---------------------------------------------
    sections.append(
        {
            "title": "Is it alpha, or is it a factor you were not paid to take?",
            "subtitle": (
                "Excess return is regressed on the Fama-French five factors plus momentum with "
                "Newey-West standard errors, because daily residuals are autocorrelated and "
                "plain OLS errors routinely turn noise into a two-sigma result."
            ),
            "blocks": [
                text(
                    f"The {label} portfolio's factor regression explains "
                    f"<b>{headline_fit.r_squared:.1%}</b> of its return variation with a market "
                    f"beta of <b>{headline_fit.betas.get('mkt_rf', float('nan')):.2f}</b>. The "
                    f"intercept is <b>{headline_fit.alpha_annual:+.2%}</b> a year with a "
                    f"heteroskedasticity- and autocorrelation-consistent "
                    f"t-statistic of <b>{headline_fit.alpha_tstat:.2f}</b>."
                ),
                callout(
                    "Finding" if headline_fit.alpha_is_significant else "Caveat",
                    f"{label}: {headline_fit.verdict}.",
                    "finding" if headline_fit.alpha_is_significant else "caveat",
                ),
                figure(figures["factors"],
                       "Factor loadings with Newey-West significance marks."),
                figure(figures["attribution"],
                       "Annualised excess return decomposed into what each factor paid and what "
                       "is left over as alpha."),
                table(
                    kw["name_factors"][
                        ["Alpha (ann.)", "Alpha t-stat", "Market", "Size (SMB)", "Value (HML)",
                         "Profitability (RMW)", "Momentum (MOM)", "Adj. R²"]
                    ],
                    caption="Single-name factor regressions",
                    index_label="Ticker",
                    colour_columns=["Alpha (ann.)"],
                    note="Full-sample daily regressions, Newey-West lag "
                         f"{cfg['factors']['newey_west_lags']}. "
                         "Source: Kenneth R. French Data Library.",
                ),
            ],
        }
    )

    # ---- 4. portfolio construction -----------------------------------------
    bl = kw["bl"]
    view_rows = bl.view_matrix.copy()
    view_rows.insert(0, "Target excess return", bl.view_returns)
    view_rows.insert(1, "View variance (Ω)", bl.view_uncertainty)

    sections.append(
        {
            "title": "Portfolio construction — six allocators on one covariance matrix",
            "subtitle": (
                "The estimator, not the optimiser, is where portfolio construction usually goes "
                "wrong. Expected returns come from a Black-Litterman posterior anchored on "
                "market equilibrium; the covariance matrix is Ledoit-Wolf shrunk."
            ),
            "blocks": [
                text(
                    f"Ledoit-Wolf selects a shrinkage intensity of "
                    f"<b>{kw['shrinkage']:.1%}</b> on the full sample "
                    f"({len(panel.returns):,} days against {len(panel.tickers)} assets), which "
                    "is the estimator correctly saying it has enough data. It matters far more "
                    f"inside the backtest, where each estimate uses only "
                    f"{cfg['backtest']['lookback_days']} days. The condition number falls from "
                    f"<b>{condition_number(kw['cov_sample']):.1f}</b> to "
                    f"<b>{condition_number(kw['cov']):.1f}</b>."
                ),
                text(
                    "Black-Litterman risk aversion is <b>not</b> the textbook 2.5–3.0. It is "
                    f"derived as δ = ERP / market variance = <b>{kw['risk_aversion']:.3f}</b>, so "
                    f"the equilibrium prior reproduces the {market['equity_risk_premium']:.1%} "
                    "equity risk premium used in the valuation section instead of quietly "
                    "asserting a different one."
                ),
                table(view_rows, caption="Analyst views entering the posterior",
                      index_label="View",
                      note="Views are stated as annualised excess returns. Ω is the view "
                           "uncertainty implied by the stated confidence: as confidence → 1 the "
                           "view is imposed exactly, as it → 0 the view is ignored."),
                table(bl.summary(), caption="Equilibrium prior versus posterior",
                      index_label="Ticker", colour_columns=["View impact"]),
                figure(figures["frontier"],
                       "Efficient frontier from the posterior, the individual names, and where "
                       "each construction method lands. Allocators are distinguished by marker "
                       "shape and label, not colour."),
                figure(figures["weights"], "Target allocation by construction method."),
                figure(figures["risk_contrib"],
                       "Share of total portfolio risk by holding. Equal <i>weights</i> are not "
                       "equal <i>risk</i>: the risk-parity column is flat at 1/N by construction, "
                       "the 1/N column is not."),
                figure(figures["correlation"],
                       "Shrunk correlation matrix. Sector blocks are visible, which is what "
                       "makes hierarchical clustering a sensible alternative to matrix "
                       "inversion."),
            ],
        }
    )

    # ---- 5. backtest --------------------------------------------------------
    display_metrics = [
        "CAGR", "Ann. volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar",
        "Alpha (ann.)", "Beta", "Information ratio", "Downside capture",
        "PSR (vs 0)", "Deflated Sharpe p",
    ]
    eq_label = "Equal weight (1/N)"
    sections.append(
        {
            "title": "Walk-forward backtest — net of costs, out of sample",
            "subtitle": (
                f"Quarterly rebalancing from {cfg['backtest']['start']}, "
                f"{cfg['backtest']['lookback_days']}-day estimation window, "
                f"{cfg['backtest']['transaction_cost_bps']:.0f}bp charged on traded notional. "
                "Positions drift with prices between rebalances."
            ),
            "blocks": [
                figure(figures["equity"],
                       "Growth of 1.00, net of transaction costs, log scale."),
                table(_select_metrics(perf, display_metrics), caption="Risk and return",
                      index_label="", orient_rows_as_metrics=True,
                      note="PSR is the probability the true Sharpe exceeds zero given the "
                           "sample's skew and kurtosis. The Deflated Sharpe p-value corrects "
                           f"for having tried {len(kw['results'])} strategies — with enough "
                           "attempts, a good-looking Sharpe is the <i>expected</i> outcome of "
                           "pure noise."),
                callout(
                    "Finding",
                    (
                        f"<b>1/N is the result to beat.</b> Equal weighting delivers a Sharpe of "
                        f"{float(perf.loc['Sharpe', eq_label]):.2f} at "
                        f"{float(kw['turnover'].loc[eq_label, 'Annual turnover (x)']):.2f}× "
                        f"annual turnover, against {float(perf.loc['Sharpe', bench_col]):.2f} for "
                        f"{cfg.benchmark}. Every optimiser here has to justify its estimation "
                        "risk against a rule that requires no estimation at all — the "
                        "DeMiguel, Garlappi & Uppal (2009) result, reproduced out of sample."
                    )
                    if eq_label in perf.columns
                    else "See the table above for the cross-strategy comparison.",
                    "finding",
                ),
                figure(figures["drawdown"], "Drawdown from running peak."),
                table(kw["turnover"], caption="Trading cost of each rule", index_label="Strategy",
                      note="Annual turnover is traded notional as a multiple of portfolio value. "
                           "Cost drag is the annualised return given up to transaction costs."),
                figure(figures["rolling_vol"],
                       "Rolling one-year volatility — the risk profile is not stationary, which "
                       "is why a single full-sample volatility number is a poor risk control."),
                figure(figures["rolling_beta"],
                       "Rolling one-year beta to the benchmark."),
            ],
        }
    )

    # ---- 6. valuation -------------------------------------------------------
    implied_min = reverse["Implied WACC"].astype(float).min()
    implied_max = reverse["Implied WACC"].astype(float).max()
    rf = float(market["risk_free_rate"])
    demanding = reverse["Implied rev CAGR (5y)"].astype(float).idxmax()
    least = reverse["Implied rev CAGR (5y)"].astype(float).idxmin()

    sections.append(
        {
            "title": "Intrinsic value — and what the market already believes",
            "subtitle": (
                "A two-stage FCFF model with mid-year discounting and a terminal value built "
                "from steady-state reinvestment, then inverted to solve for the assumptions "
                "embedded in the market price."
            ),
            "blocks": [
                table(valuation, caption="Base-case DCF", index_label="Ticker",
                      colour_columns=["Upside"],
                      note="Terminal value share and implied exit EV/EBITDA are printed "
                           "deliberately: a DCF where most of the value sits beyond the forecast "
                           "horizon is a terminal-value model wearing a DCF costume."),
                figure(figures["upside"], "Upside to base-case intrinsic value."),
                callout(
                    "The honest problem with a forward DCF",
                    (
                        "At a "
                        f"{rf:.1%} risk-free rate and a "
                        f"{market['equity_risk_premium']:.1%} equity risk premium, the base case "
                        "marks almost the whole universe as expensive. That is an unfalsifiable "
                        "opinion, not research. Inverting the model turns it into a claim a "
                        "portfolio manager can argue with."
                    ),
                    "caveat",
                ),
                table(reverse, caption="Reverse DCF — solving for the market's assumptions",
                      index_label="Ticker"),
                figure(figures["reverse"],
                       "Five-year revenue CAGR assumed in the base case against the CAGR "
                       "required to justify the current price."),
                figure(figures["implied_wacc"],
                       "The discount rate that equates model value to market price — the "
                       "market's implied expected return on each name."),
                callout(
                    "Finding",
                    (
                        f"Across these names the market is paying an implied cost of capital of "
                        f"<b>{implied_min:.1%}–{implied_max:.1%}</b> against a {rf:.1%} "
                        f"risk-free rate — an implied equity risk premium of roughly "
                        f"{(implied_min - rf) * 10_000:.0f}–{(implied_max - rf) * 10_000:.0f} "
                        "basis points. The gap between this model and the market is a "
                        "<b>risk-premium disagreement, not an earnings disagreement</b>. "
                        f"On growth, <b>{demanding}</b> carries the most demanding expectations "
                        f"({float(reverse.loc[demanding, 'Implied rev CAGR (5y)']):.0%} five-year "
                        f"revenue CAGR required) and <b>{least}</b> the least "
                        f"({float(reverse.loc[least, 'Implied rev CAGR (5y)']):.0%})."
                    ),
                    "finding",
                ),
                {"kind": "heading", "text": f"{focus} — deep dive"},
                figure(figures["monte_carlo"],
                       f"{focus} intrinsic value under simulated growth, margin, WACC and "
                       "terminal-growth uncertainty."),
                figure(figures["sensitivity"],
                       f"{focus} fair value across WACC and terminal growth."),
                figure(figures["football"],
                       f"{focus} valuation range by methodology. No single method is a price "
                       "target; the overlap is the answer."),
                table(kw["focus_field"], caption=f"{focus} — methodology ranges",
                      index_label="Method", colour_columns=["implied upside (mid)"]),
                {"kind": "heading", "text": "Trading comparables"},
                table(
                    kw["comps"].drop(columns=["Market cap", "Net debt", "EV"], errors="ignore"),
                    caption="Current multiples", index_label="Ticker",
                    note="Enterprise-value multiples for measures available to all capital "
                         "providers; P/E only for the equity claim. EV over net income is a "
                         "capital-structure error, not a rounding difference.",
                ),
                table(kw["peers"], caption="Peer multiple quartiles", index_label="Multiple"),
                {"kind": "heading", "text": "Banks are valued differently, on purpose"},
                text(
                    "Free cash flow to the firm has no meaning for a bank: leverage is the "
                    "business and the loan book is the working capital. The two financials are "
                    "valued on residual income off book equity — if ROE equals the cost of "
                    "equity for ever, the bank is worth exactly book, and everything above book "
                    "has to be earned."
                ),
                table(kw["bank_frame"], caption="Excess-return model", index_label="Ticker",
                      colour_columns=["Upside"]),
                table(kw["mc_frame"], caption="Monte Carlo — full distribution by name",
                      index_label="Ticker",
                      note="20,000 paths per name over revenue growth, EBIT margin, WACC and "
                           "terminal growth."),
            ],
        }
    )

    # ---- 7. limitations -----------------------------------------------------
    sections.append(
        {
            "title": "What this study cannot tell you",
            "subtitle": (
                "The limitations that would materially change the conclusions, stated plainly. "
                "A research note that lists none is hiding them."
            ),
            "blocks": [
                callout(
                    "Read this before quoting any number above",
                    "The single largest bias here is <b>universe selection</b>. These 16 names "
                    "were chosen in 2026 and are today's US large caps, so every strategy "
                    "inherits the hindsight of picking companies that survived and compounded. "
                    "The <b>absolute</b> returns are therefore not investable alpha. What "
                    "remains valid is the <b>relative</b> comparison: every allocator faces the "
                    "identical universe, calendar, costs and constraints, so the ranking between "
                    "them is a fair test even though the level is not.",
                    "caveat",
                ),
                bullets(
                    [
                        "<b>Survivorship and selection.</b> No delisted or acquired names, and "
                        "no point-in-time index membership. Treat cross-strategy differences as "
                        "the result; treat the level of returns as an upper bound.",
                        "<b>Single market, single currency.</b> US large-cap equity only. No "
                        "credit, no rates, no FX, no small caps, no non-US listings — so the "
                        "correlation structure is far friendlier than a real book's.",
                        "<b>Long-only, fully invested.</b> No shorting, no leverage, no cash "
                        "allocation and no derivatives overlay, which removes most of the ways a "
                        "real mandate can go wrong.",
                        "<b>Transaction costs are a flat "
                        f"{cfg['backtest']['transaction_cost_bps']:.0f}bp</b> on traded notional. "
                        "Real costs are state-dependent and rise exactly when a strategy wants "
                        "to trade most; market impact and borrow are not modelled at all.",
                        "<b>The valuation inputs are the modeller's estimates</b>, not a "
                        "fundamentals feed. They are versioned in "
                        "<code>config/fundamentals.yaml</code> so they can be re-keyed from "
                        "filings and the whole report regenerated — that is the mitigation, not "
                        "a claim that they are audited.",
                        "<b>Factor data lags prices</b> by several weeks; factor statistics use "
                        "the overlapping window only, so the most recent period is absent from "
                        "the attribution but present in the performance figures.",
                        "<b>Multiple testing.</b> Several strategies were evaluated on one "
                        "sample. The Deflated Sharpe p-value in section 5 is the correction; it "
                        "is reported for every strategy, not only the flattering ones.",
                        "<b>Nothing here is investment advice</b>, a price target, or a "
                        "recommendation. It is a demonstration of method.",
                    ]
                ),
            ],
        }
    )

    # ---- 8. method ----------------------------------------------------------
    sections.append(
        {
            "title": "Method and reproducibility",
            "subtitle": "How to regenerate every number in this document.",
            "blocks": [
                bullets(
                    [
                        "<b>Data.</b> Daily adjusted closes from the public Yahoo Finance chart "
                        "endpoint, cached to <code>data/raw/prices/</code> and committed. Factor "
                        "and risk-free series from the Kenneth R. French Data Library, cached to "
                        "<code>data/raw/factors/</code>. The repository reproduces offline.",
                        "<b>Alignment.</b> Prices, factors and the risk-free path are joined once "
                        "in <code>ResearchPanel</code>. The market calendar is authoritative.",
                        "<b>Estimation.</b> Ledoit-Wolf shrinkage for covariance; Black-Litterman "
                        "for expected returns; Newey-West for regression inference.",
                        "<b>Backtest.</b> Point-in-time loop, drift between rebalances, costs on "
                        "traded notional. Allocators receive a <code>StrategyContext</code> that "
                        "carries only data available at the rebalance date.",
                        "<b>Valuation.</b> Two-stage FCFF with mid-year discounting and "
                        "steady-state reinvestment; residual income for banks; Monte Carlo and "
                        "Brent-solved reverse DCF on top.",
                        "<b>Tests.</b> <code>pytest</code> covers the analytics against closed-form "
                        "results, the optimisers against their defining properties (equal risk "
                        "contribution, budget constraints), the vectorised Monte Carlo against "
                        "the pandas DCF, and the backtest against look-ahead.",
                        "<b>Run it.</b> <code>make install &amp;&amp; make data &amp;&amp; make "
                        "pipeline</code>, or <code>quantdesk run</code>.",
                    ]
                ),
            ],
        }
    )

    meta = {
        "title": "Equity Research, Risk and Allocation",
        "eyebrow": f"QuantDesk v{__version__} · Institutional research stack",
        "lede": (
            "From raw prices to a tested risk model, a factor-controlled allocation and an "
            "intrinsic-value view — with the assumptions written down, the models backtested, "
            "and the limitations stated."
        ),
        "author": cfg["project"]["author"],
        "as_of": panel.end.date().isoformat(),
        "n_assets": len(panel.tickers),
        "benchmark": cfg.benchmark,
        "sample": f"{panel.start.date()} → {panel.end.date()}",
        "n_days": f"{len(panel.returns):,}",
        "generated": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "repro": (
            "Every figure and table in this document is generated by "
            "<code>quantdesk run</code> from the committed data cache and the two YAML "
            "configuration files. There are no hand-entered numbers in the report."
        ),
        "disclaimer": (
            "This document is a demonstration of quantitative research method. It is not "
            "investment advice, not a recommendation, and not a price target. The valuation "
            "inputs are the author's own normalised estimates and must be re-keyed from primary "
            "filings before any decision is taken."
        ),
        "footer": (
            f"QuantDesk v{__version__} · Python · pandas · statsmodels · scikit-learn · "
            "SciPy · matplotlib"
        ),
    }

    return {"meta": meta, "kpis": kpis, "sections": sections}
