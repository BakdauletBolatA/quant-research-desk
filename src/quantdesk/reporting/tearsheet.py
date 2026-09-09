"""HTML research report.

One self-contained file: images are embedded as data URIs so the report can be
emailed, opened offline, or attached to an application without a folder of
loose PNGs going missing.
"""

from __future__ import annotations

import base64
import datetime as dt
import html
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).parent / "templates"

_PERCENT_HINTS = (
    "return", "cagr", "vol", "drawdown", "upside", "yield", "rate", "weight", "alpha",
    "growth", "var ", "es ", "share", "capture", "roe", "roic", "wacc", "premium",
    "hit", "drag", "deflated", "sharpe p", "cost_of", "cost of", "terminal g", "erp", "contribution", "exposure", "p(", "psr", "dsr", "shrinkage", "tracking",
    "ulcer", "margin",
)
_INTEGER_HINTS = ("obs", "days", "count", "exceptions", "rebalances", "paths", "positions")
_SIGNED_HINTS = ("upside", "alpha", "impact", "gap", "active")


_PERCENT_EXCLUSIONS = ("shares", "count", "number", "paths", "ratio")


def _is_percent(label: str) -> bool:
    """Heuristic number format. Explicit exclusions come first: `shares_diluted`
    contains "share" but is a count, and rendering it as 745,000% is exactly the
    kind of silent formatting bug that discredits an otherwise correct model."""
    name = str(label).lower()
    if any(bad in name for bad in _PERCENT_EXCLUSIONS):
        return False
    return any(hint in name for hint in _PERCENT_HINTS)


def _is_integer(label: str) -> bool:
    name = str(label).lower()
    return any(hint in name for hint in _INTEGER_HINTS)


def format_value(value: Any, label: str, decimals: int = 2) -> str:
    """Format a single cell the way a research note would print it."""
    if value is None:
        return "—"
    if isinstance(value, str):
        return value
    if isinstance(value, (pd.Timestamp, dt.date, dt.datetime, np.datetime64)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return "—"
        if _is_integer(label):
            return f"{number:,.0f}"
        if _is_percent(label):
            signed = any(hint in str(label).lower() for hint in _SIGNED_HINTS)
            return f"{number:{'+' if signed else ''}.2%}"
        if abs(number) >= 10_000:
            return f"{number:,.0f}"
        return f"{number:,.{decimals}f}"
    return str(value)


def frame_to_html(
    frame: pd.DataFrame,
    *,
    index_label: str = "",
    orient_rows_as_metrics: bool = False,
    colour_columns: Sequence[str] | None = None,
    decimals: int = 2,
) -> str:
    """Render a DataFrame as a styled table.

    ``orient_rows_as_metrics`` switches the formatting key from the column name
    to the row name, which is what a stats-down/entities-across table needs.
    """
    colour_columns = set(colour_columns or ())
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []

    for index_value, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            label = str(index_value) if orient_rows_as_metrics else str(column)
            value = row[column]
            text = html.escape(format_value(value, label, decimals))

            css = ""
            if isinstance(value, str):
                upper = value.strip().upper()
                if upper in ("BUY", "SELL", "HOLD"):
                    css = f' class="tag-{upper.lower()}"'
            elif (
                column in colour_columns
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(float(value))
            ):
                css = ' class="pos"' if float(value) >= 0 else ' class="neg"'
            cells.append(f"<td{css}>{text}</td>")
        rows.append(f"<tr><th>{html.escape(str(index_value))}</th>{''.join(cells)}</tr>")

    return (
        f"<table><thead><tr><th>{html.escape(index_label)}</th>{head}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def embed_image(path: str | Path) -> str:
    """Inline a PNG as a data URI so the report is one portable file."""
    data = Path(path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------------------
# Block builders — small helpers so the pipeline reads like an outline
# ---------------------------------------------------------------------------
def text(markup: str) -> dict[str, Any]:
    return {"kind": "text", "html": markup}


def heading(label: str) -> dict[str, Any]:
    return {"kind": "heading", "text": label}


def callout(tag: str, markup: str, variant: str = "") -> dict[str, Any]:
    return {"kind": "callout", "tag": tag, "html": markup, "variant": variant}


def bullets(items: Iterable[str]) -> dict[str, Any]:
    return {"kind": "list", "entries": list(items)}


def table(frame: pd.DataFrame, caption: str = "", note: str = "", **kwargs) -> dict[str, Any]:
    return {
        "kind": "table",
        "caption": caption,
        "note": note,
        "html": frame_to_html(frame, **kwargs),
    }


class FigureCounter:
    """Sequential figure numbering across the whole report."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, path: str | Path, caption: str) -> dict[str, Any]:
        self.count += 1
        return {
            "kind": "figure",
            "src": embed_image(path),
            "caption": caption,
            "number": self.count,
        }


def render_tearsheet(
    output_path: str | Path,
    *,
    meta: dict[str, Any],
    kpis: list[dict[str, str]],
    sections: list[dict[str, Any]],
    template_name: str = "tearsheet.html.jinja",
) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(enabled_extensions=("jinja",), default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    rendered = env.get_template(template_name).render(meta=meta, kpis=kpis, sections=sections)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return str(output_path)
