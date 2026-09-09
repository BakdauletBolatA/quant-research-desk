"""Report rendering: formatting rules, chart output, workbook and HTML."""

from __future__ import annotations

import itertools

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from quantdesk.reporting import charts
from quantdesk.reporting.excel import write_workbook
from quantdesk.reporting.tearsheet import (
    format_value,
    frame_to_html,
    render_tearsheet,
    table,
    text,
)


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "label,value,expected",
    [
        ("CAGR", 0.2204, "22.04%"),
        ("Ann. volatility", 0.1752, "17.52%"),
        ("Max drawdown", -0.3385, "-33.85%"),
        ("Upside", 0.1343, "+13.43%"),          # signed columns carry the sign
        ("Alpha (ann.)", -0.02, "-2.00%"),
        ("Sharpe", 1.104, "1.10"),
        ("Kupiec p", 0.4758, "0.48"),           # p-values stay decimal, by convention
        ("Obs", 3191.0, "3,191"),
        ("Exceptions", 36, "36"),
        ("shares_diluted", 14700.0, "14,700"),  # a count, never a percentage
        ("Terminal g", 0.03, "3.00%"),
        ("Signal", "BUY", "BUY"),
        ("Sharpe", float("nan"), "—"),
        ("Sharpe", None, "—"),
    ],
)
def test_format_value(label, value, expected):
    assert format_value(value, label) == expected


def test_shares_are_never_rendered_as_a_percentage():
    """Guards the exact bug that would print 14,700 shares as 1,470,000%."""
    frame = pd.DataFrame({"shares_diluted": [14700.0], "tax_rate": [0.24]}, index=["AAPL"])
    html = frame_to_html(frame)
    assert "14,700" in html
    assert "24.00%" in html
    assert "1,470,000.00%" not in html


def test_frame_to_html_marks_signs_and_signals():
    frame = pd.DataFrame(
        {"Upside": [0.2, -0.3], "Signal": ["BUY", "SELL"]}, index=["A", "B"]
    )
    html = frame_to_html(frame, index_label="Ticker", colour_columns=["Upside"])
    assert '<td class="pos">+20.00%</td>' in html
    assert '<td class="neg">-30.00%</td>' in html
    assert 'class="tag-buy"' in html and 'class="tag-sell"' in html
    assert "<th>Ticker</th>" in html


def test_frame_to_html_escapes_untrusted_text():
    frame = pd.DataFrame({"Note": ["<script>alert(1)</script>"]}, index=["X"])
    html = frame_to_html(frame)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_metric_orientation_switches_the_formatting_key():
    """Stats down the side, entities across: the row name decides the format."""
    frame = pd.DataFrame({"Strategy A": [0.22, 1.10]}, index=["CAGR", "Sharpe"])
    html = frame_to_html(frame, orient_rows_as_metrics=True)
    assert "22.00%" in html
    assert "1.10" in html


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
@pytest.fixture
def chart_inputs(synthetic_returns, synthetic_market):
    return synthetic_returns, synthetic_market.rename("MKT")


def test_every_chart_writes_a_non_trivial_png(tmp_path, chart_inputs, synthetic_series):
    returns, market = chart_inputs
    written: list[str] = []

    written.append(charts.equity_curves(returns, tmp_path / "a.png", benchmark=market))
    written.append(charts.drawdown_chart(returns, tmp_path / "b.png"))
    written.append(
        charts.rolling_metric({"vol": returns["HIGH"].rolling(60).std()},
                              tmp_path / "c.png", "Rolling", "Vol", as_percent=True)
    )
    written.append(charts.correlation_heatmap(returns.corr(), tmp_path / "d.png"))
    written.append(
        charts.monte_carlo_distribution(
            np.random.default_rng(0).normal(100, 12, 5000), tmp_path / "e.png",
            "TEST", 95.0, 101.0,
        )
    )
    written.append(
        charts.upside_chart(pd.DataFrame({"Upside": [0.2, -0.1]}, index=["A", "B"]),
                            tmp_path / "f.png")
    )
    written.append(
        charts.attribution_chart(
            pd.Series({"Market": 0.09, "Alpha": 0.02, "Total explained": 0.11}),
            tmp_path / "g.png", "Attribution",
        )
    )
    written.append(
        charts.factor_exposures(
            pd.Series({"mkt_rf": 1.1, "smb": -0.3}), pd.Series({"mkt_rf": 30.0, "smb": -2.2}),
            tmp_path / "h.png", "Loadings",
        )
    )

    for path in written:
        assert (tmp_path / path.rsplit("/", 1)[-1]).stat().st_size > 5_000, path


def test_var_charts_render(tmp_path, synthetic_series):
    from quantdesk.analytics.risk import rolling_var_forecast, var_backtest_table

    path = rolling_var_forecast(synthetic_series, 0.99, "historical", window=500)
    out = charts.var_exceptions(synthetic_series, path, tmp_path / "var.png", 0.99,
                                "historical")
    assert (tmp_path / "var.png").stat().st_size > 5_000 and out

    table_ = var_backtest_table(synthetic_series, 0.99, window=500)
    charts.var_model_comparison(table_, tmp_path / "models.png", 0.01)
    assert (tmp_path / "models.png").stat().st_size > 5_000


def test_football_field_and_sensitivity_render(tmp_path):
    field = pd.DataFrame(
        {"low": [80.0, 70.0], "high": [120.0, 140.0], "mid": [100.0, 105.0]},
        index=["DCF", "Peers"],
    )
    charts.football_field_chart(field, tmp_path / "ff.png", "TEST", 95.0)
    assert (tmp_path / "ff.png").stat().st_size > 5_000

    grid = pd.DataFrame(
        np.arange(25, dtype=float).reshape(5, 5) + 80,
        index=np.linspace(0.07, 0.11, 5), columns=np.linspace(0.015, 0.035, 5),
    )
    charts.sensitivity_heatmap(grid, tmp_path / "sens.png", "TEST", 95.0)
    assert (tmp_path / "sens.png").stat().st_size > 5_000


def test_spread_labels_preserves_order_and_separates():
    positions = [1.0, 1.0001, 1.0002, 5.0]
    spread = charts._spread_labels(positions, min_gap=0.5)
    assert spread == sorted(spread)
    assert all(b - a >= 0.5 - 1e-9 for a, b in itertools.pairwise(spread))


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------
def test_workbook_round_trips(tmp_path):
    frame = pd.DataFrame(
        {"CAGR": [0.22, 0.14], "shares_diluted": [14700.0, 7450.0], "Signal": ["BUY", "HOLD"]},
        index=["AAPL", "MSFT"],
    )
    path = write_workbook(
        tmp_path / "book.xlsx", {"Summary": [("Results", frame)]},
        title="Test", subtitle="Sub", notes={"Summary": "A note."},
    )
    assert (tmp_path / "book.xlsx").stat().st_size > 4_000

    read_back = pd.read_excel(path, sheet_name="Summary", header=None)
    flat = [str(cell) for cell in read_back.to_numpy().ravel()]
    assert any("Test" in cell for cell in flat)
    assert any("AAPL" in cell for cell in flat)


def test_workbook_skips_empty_tables(tmp_path):
    path = write_workbook(
        tmp_path / "b.xlsx",
        {"S": [("Empty", pd.DataFrame()), ("Real", pd.DataFrame({"x": [1]}, index=["a"]))]},
        title="T", subtitle="S",
    )
    assert pd.read_excel(path, sheet_name="S", header=None).shape[0] > 3


def test_tearsheet_renders_a_self_contained_document(tmp_path):
    frame = pd.DataFrame({"CAGR": [0.22]}, index=["Strategy"])
    output = render_tearsheet(
        tmp_path / "report.html",
        meta={
            "title": "T", "eyebrow": "E", "lede": "L", "author": "A", "as_of": "2026-09-09",
            "n_assets": 16, "benchmark": "SPY", "sample": "S", "n_days": "3,691",
            "generated": "now", "repro": "R", "disclaimer": "D", "footer": "F",
        },
        kpis=[{"label": "K", "value": "1.10", "foot": "f", "direction": "up"}],
        sections=[
            {
                "title": "Section one",
                "subtitle": "Sub",
                "blocks": [text("Some <b>markup</b>."), table(frame, caption="Cap")],
            }
        ],
    )
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert output
    assert "<!doctype html>" in html.lower()
    assert "Section one" in html and "22.00%" in html
    assert "Some <b>markup</b>." in html, "trusted narrative markup must survive"
    assert "http://" not in html and "https://" not in html, "no external dependencies"
