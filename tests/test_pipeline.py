"""End-to-end smoke test.

Runs the real pipeline on the committed data cache and checks that the
deliverables exist, contain what they claim, and are reproducible. This is the
test that would catch an integration break the unit tests cannot see.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from quantdesk.cli import build_parser, main
from quantdesk.config import RAW_DIR, load_config

CACHE_READY = (RAW_DIR / "factors" / "ff5_mom_daily.csv").exists() and any(
    (RAW_DIR / "prices").glob("*.csv")
)
pytestmark = pytest.mark.skipif(not CACHE_READY, reason="data cache not populated")


@pytest.fixture(scope="module")
def result():
    from quantdesk.pipeline import run_pipeline

    return run_pipeline(load_config())


def test_deliverables_are_written(result):
    from pathlib import Path

    assert Path(result.report_html).stat().st_size > 200_000
    assert Path(result.workbook).stat().st_size > 10_000
    assert len(result.figures) >= 18
    for path in result.figures.values():
        assert Path(path).stat().st_size > 5_000, path


def test_report_contains_the_argument_not_just_numbers(result):
    from pathlib import Path

    html = Path(result.report_html).read_text(encoding="utf-8")
    for phrase in (
        "What this study cannot tell you",
        "universe selection",
        "Kupiec",
        "Newey-West",
        "Reverse DCF",
        "Deflated Sharpe",
        "not investment advice",
    ):
        assert phrase in html, phrase
    assert html.count("data:image/png;base64,") >= 18
    # A missing value must render as an em dash, never leak a repr. Checked on
    # whole table cells so that words such as "financials" do not false-positive.
    leaked = re.findall(r">\s*(?:nan|NaN|None|inf|-inf|<NA>)\s*<", html)
    assert not leaked, leaked


def test_every_strategy_produced_a_track_record(result):
    configured = load_config()["backtest"]["strategies"]
    assert set(result.backtests) == set(configured)
    for backtest in result.backtests.values():
        assert len(backtest.returns) > 250
        assert np.isfinite(backtest.returns.to_numpy()).all()


def test_performance_table_has_no_missing_headline_metrics(result):
    for metric in ("CAGR", "Sharpe", "Max drawdown", "Deflated Sharpe p"):
        assert metric in result.performance.index
    core = result.performance.loc[["CAGR", "Sharpe", "Max drawdown"]]
    assert core.notna().to_numpy().all()


def test_valuation_and_reverse_tables_agree_on_coverage(result):
    assert list(result.valuation.index) == list(result.reverse.index)
    assert (result.reverse["Implied WACC"].astype(float) > 0).all()
    assert result.valuation["Upside"].astype(float).notna().all()


def test_var_validation_ran_every_estimator(result):
    from quantdesk.analytics.risk import VAR_METHODS

    assert set(result.var_validation.index) == set(VAR_METHODS)
    assert (result.var_validation["observations"] > 1000).all()


def test_weights_are_valid_portfolios(result):
    sums = result.weights.sum(axis=0)
    assert np.allclose(sums.to_numpy(), 1.0, atol=1e-6)
    assert (result.weights.to_numpy() >= -1e-9).all()


def test_pipeline_is_reproducible(result):
    from quantdesk.pipeline import run_pipeline

    again = run_pipeline(load_config())
    assert again.performance.loc["Sharpe"].round(10).equals(
        result.performance.loc["Sharpe"].round(10)
    )
    assert np.allclose(
        again.valuation["Fair value"].astype(float).to_numpy(),
        result.valuation["Fair value"].astype(float).to_numpy(),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def test_parser_exposes_every_command():
    parser = build_parser()
    for command in ("data", "run", "backtest", "value"):
        assert parser.parse_args([command] if command != "value" else [command, "AAPL"])


def test_value_command_runs_for_an_industrial_and_a_bank(capsys):
    assert main(["value", "MSFT"]) == 0
    output = capsys.readouterr().out
    assert "Reverse DCF" in output and "Monte Carlo" in output

    assert main(["value", "JPM"]) == 0
    assert "residual income" in capsys.readouterr().out


def test_value_command_rejects_an_unknown_ticker(capsys):
    assert main(["value", "ZZZZ"]) == 1
    assert "not in the universe" in capsys.readouterr().out
