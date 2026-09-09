"""Command line interface.

    quantdesk data              refresh the price and factor cache
    quantdesk run               run the full study and write reports/
    quantdesk backtest          print the walk-forward comparison table
    quantdesk value AAPL        print one company's valuation, base and inverted
"""

from __future__ import annotations

import argparse
import logging
import sys

import matplotlib

matplotlib.use("Agg")  # noqa: E402 - report generation is headless

import pandas as pd  # noqa: E402

from quantdesk import __version__  # noqa: E402
from quantdesk.config import Config, ensure_dirs, load_config, load_fundamentals  # noqa: E402


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _banner(cfg: Config) -> None:
    project = cfg["project"]
    print(f"\n\033[1m{project['name']} v{__version__}\033[0m — {project['subtitle']}")
    print(f"{'─' * 78}\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def cmd_data(args: argparse.Namespace) -> int:
    from quantdesk.data import build_panel

    cfg = load_config(args.config)
    _banner(cfg)
    ensure_dirs()
    panel = build_panel(cfg, force=args.force)
    print(panel.summary().to_string(float_format=lambda v: f"{v:,.4f}"))
    print(f"\n{len(panel.returns):,} trading days · {panel.start.date()} → {panel.end.date()}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from quantdesk.pipeline import run_pipeline

    cfg = load_config(args.config)
    _banner(cfg)
    result = run_pipeline(cfg, force_download=args.force)

    print("\n\033[1mStrategy comparison (net of costs)\033[0m")
    metrics = ["CAGR", "Ann. volatility", "Sharpe", "Max drawdown", "Deflated Sharpe p"]
    rows = [m for m in metrics if m in result.performance.index]
    print(result.performance.loc[rows].T.to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n\033[1mDeliverables\033[0m")
    print(f"  report    {result.report_html}")
    print(f"  workbook  {result.workbook}")
    print(f"  figures   {len(result.figures)} PNGs in reports/figures/\n")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    from quantdesk.analytics import performance_summary
    from quantdesk.backtest import combine_returns, run_all, turnover_table
    from quantdesk.data import build_panel
    from quantdesk.portfolio import Constraints

    cfg = load_config(args.config)
    fundamentals = load_fundamentals()
    _banner(cfg)

    panel = build_panel(cfg)
    backtest_cfg, portfolio_cfg = cfg["backtest"], cfg["portfolio"]
    results = run_all(
        backtest_cfg["strategies"], panel.returns, panel.prices, panel.risk_free,
        start=backtest_cfg["start"], rebalance=backtest_cfg["rebalance"],
        lookback_days=backtest_cfg["lookback_days"],
        transaction_cost_bps=backtest_cfg["transaction_cost_bps"],
        constraints=Constraints(
            portfolio_cfg["long_only"], portfolio_cfg["min_weight"], portfolio_cfg["max_weight"]
        ),
        covariance_method=portfolio_cfg["covariance"],
        periods_per_year=panel.periods_per_year,
        shares_outstanding=fundamentals["shares_outstanding"],
        equity_risk_premium=fundamentals["market"]["equity_risk_premium"],
        views=portfolio_cfg["black_litterman"]["views"],
        tau=portfolio_cfg["black_litterman"]["tau"],
        risk_aversion=portfolio_cfg.get("risk_aversion"),
    )
    returns = combine_returns(results)
    benchmark = panel.benchmark_returns.reindex(returns.index)
    returns[f"{cfg.benchmark} (benchmark)"] = benchmark

    summary = performance_summary(returns, benchmark, panel.risk_free, panel.periods_per_year)
    rows = ["CAGR", "Ann. volatility", "Sharpe", "Sortino", "Max drawdown", "Calmar",
            "Alpha (ann.)", "Information ratio"]
    print(summary.loc[[r for r in rows if r in summary.index]].T.to_string(
        float_format=lambda v: f"{v:,.3f}"))
    print()
    print(turnover_table(results).to_string(float_format=lambda v: f"{v:,.3f}"))
    return 0


def cmd_value(args: argparse.Namespace) -> int:
    from quantdesk.data import build_panel
    from quantdesk.valuation import run_dcf, run_excess_return_model, run_monte_carlo
    from quantdesk.valuation.reverse_dcf import reverse_dcf

    cfg = load_config(args.config)
    fundamentals = load_fundamentals()
    _banner(cfg)

    ticker = args.ticker.upper()
    panel = build_panel(cfg)
    if ticker not in panel.prices.columns:
        print(f"{ticker} is not in the universe: {', '.join(panel.tickers)}")
        return 1
    price = float(panel.prices[ticker].iloc[-1])
    market = fundamentals["market"]

    if ticker in fundamentals["excess_return"]:
        result = run_excess_return_model(
            ticker, fundamentals["excess_return"][ticker], market=market, current_price=price
        )
        print(pd.Series(result.to_row()).to_string())
        print("\nModel: residual income on book equity (banks are not valued with FCFF).")
        return 0

    if ticker not in fundamentals["dcf"]:
        print(f"No valuation inputs for {ticker} in config/fundamentals.yaml")
        return 1

    inputs = fundamentals["dcf"][ticker]
    base = run_dcf(ticker, inputs, market=market, current_price=price)
    print(pd.Series(base.to_row()).to_string())

    print("\n\033[1mProjection\033[0m")
    print(base.projection[["revenue", "ebit", "nopat", "fcff", "pv_fcff"]].to_string(
        float_format=lambda v: f"{v:,.0f}"))

    mc = run_monte_carlo(
        ticker, inputs, market=market, current_price=price, base_wacc=base.wacc,
        base_value=base.value_per_share, mc_config=fundamentals["monte_carlo"],
        n_paths=int(cfg["valuation"]["monte_carlo_paths"]), seed=int(cfg["valuation"]["seed"]),
    )
    print("\n\033[1mMonte Carlo\033[0m")
    print(mc.to_series().to_string(float_format=lambda v: f"{v:,.2f}"))

    inverted = reverse_dcf(ticker, inputs, market=market, current_price=price, base_result=base)
    print("\n\033[1mReverse DCF\033[0m")
    print(inverted.narrative())
    return 0


# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantdesk",
        description="Institutional equity research, valuation and portfolio risk platform.",
    )
    parser.add_argument("--version", action="version", version=f"quantdesk {__version__}")
    parser.add_argument("-c", "--config", default=None, help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore the local data cache")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("data", help="download and cache prices and factors").set_defaults(
        func=cmd_data
    )
    sub.add_parser("run", help="run the full study and write reports/").set_defaults(func=cmd_run)
    sub.add_parser("backtest", help="print the walk-forward comparison").set_defaults(
        func=cmd_backtest
    )
    value = sub.add_parser("value", help="value a single company")
    value.add_argument("ticker")
    value.set_defaults(func=cmd_value)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("quantdesk").error("%s: %s", type(exc).__name__, exc)
        if args.verbose:
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
