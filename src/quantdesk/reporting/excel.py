"""Excel deliverable.

The audience for this file is not the person who ran the code — it is the
colleague who wants to check a number, and the interviewer who asks "can I see
the working?". So every sheet is formatted, frozen and labelled, percentages
are percentages rather than 0.0734, and the assumptions live on their own sheet
next to the outputs they drive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# Column-name fragments that decide the number format when none is given.
_PERCENT_HINTS = (
    "return", "cagr", "vol", "drawdown", "upside", "yield", "rate", "weight", "alpha",
    "growth", "var ", "es ", "share", "capture", "roe", "roic", "wacc", "premium",
    "hit", "drag", "contribution", "exposure", "p(", "psr", "dsr", "shrinkage",
)
_INTEGER_HINTS = ("obs", "days", "count", "exceptions", "rebalances", "paths", "n ")


def _format_for(column: str, formats: dict[str, Any]) -> Any:
    name = str(column).lower()
    if any(hint in name for hint in _INTEGER_HINTS):
        return formats["int"]
    if any(hint in name for hint in _PERCENT_HINTS):
        return formats["pct"]
    return formats["num"]


def _build_formats(workbook) -> dict[str, Any]:
    base = {"font_name": "Calibri", "font_size": 10}
    return {
        "title": workbook.add_format(
            {**base, "bold": True, "font_size": 14, "font_color": "#0b0b0b"}
        ),
        "subtitle": workbook.add_format({**base, "font_size": 10, "font_color": "#52514e"}),
        "section": workbook.add_format(
            {**base, "bold": True, "font_size": 11, "font_color": "#2a78d6"}
        ),
        "header": workbook.add_format(
            {
                **base, "bold": True, "font_color": "#ffffff", "bg_color": "#2a78d6",
                "border": 1, "border_color": "#2a78d6", "align": "center", "valign": "vcenter",
                "text_wrap": True,
            }
        ),
        "index": workbook.add_format({**base, "bold": True, "bg_color": "#f0efec", "border": 1,
                                      "border_color": "#e1e0d9"}),
        "num": workbook.add_format({**base, "num_format": "#,##0.00", "border": 1,
                                    "border_color": "#e1e0d9"}),
        "int": workbook.add_format({**base, "num_format": "#,##0", "border": 1,
                                    "border_color": "#e1e0d9"}),
        "pct": workbook.add_format({**base, "num_format": "0.00%", "border": 1,
                                    "border_color": "#e1e0d9"}),
        "text": workbook.add_format({**base, "border": 1, "border_color": "#e1e0d9"}),
        "date": workbook.add_format({**base, "num_format": "yyyy-mm-dd", "border": 1,
                                     "border_color": "#e1e0d9"}),
        "note": workbook.add_format({**base, "italic": True, "font_color": "#898781",
                                     "text_wrap": True, "valign": "top"}),
    }


def _write_table(
    worksheet,
    frame: pd.DataFrame,
    row: int,
    formats: dict[str, Any],
    index_label: str = "",
    percent_columns: Sequence[str] | None = None,
) -> int:
    """Write one DataFrame with a header row and a formatted index column."""
    percent_columns = set(percent_columns or ())

    worksheet.write(row, 0, index_label, formats["header"])
    for j, column in enumerate(frame.columns, start=1):
        worksheet.write(row, j, str(column), formats["header"])

    for i, (label, series) in enumerate(frame.iterrows(), start=row + 1):
        worksheet.write(i, 0, str(label), formats["index"])
        for j, column in enumerate(frame.columns, start=1):
            value = series[column]
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                worksheet.write_datetime(i, j, pd.Timestamp(value).to_pydatetime(),
                                         formats["date"])
            elif value is None or (isinstance(value, float) and not np.isfinite(value)):
                worksheet.write(i, j, "n/a", formats["text"])
            elif isinstance(value, (int, float, np.integer, np.floating)):
                fmt = formats["pct"] if column in percent_columns else _format_for(column, formats)
                worksheet.write_number(i, j, float(value), fmt)
            else:
                worksheet.write(i, j, str(value), formats["text"])

    widest_index = max([len(str(i)) for i in frame.index] + [len(index_label)]) + 2
    worksheet.set_column(0, 0, min(max(widest_index, 12), 40))
    worksheet.set_column(1, len(frame.columns), 15)
    return row + len(frame) + 2


def write_workbook(
    path: Path,
    sheets: dict[str, list[tuple[str, pd.DataFrame]]],
    *,
    title: str,
    subtitle: str,
    notes: dict[str, str] | None = None,
) -> str:
    """Write the full research workbook.

    ``sheets`` maps a sheet name to an ordered list of (section title, table).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = notes or {}

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        formats = _build_formats(workbook)

        for sheet_name, tables in sheets.items():
            worksheet = workbook.add_worksheet(sheet_name[:31])
            writer.sheets[sheet_name[:31]] = worksheet
            worksheet.hide_gridlines(2)
            worksheet.set_row(0, 22)

            worksheet.write(0, 0, title, formats["title"])
            worksheet.write(1, 0, subtitle, formats["subtitle"])
            row = 3

            if sheet_name in notes:
                worksheet.merge_range(row, 0, row, 7, notes[sheet_name], formats["note"])
                worksheet.set_row(row, 32)
                row += 2

            for section_title, frame in tables:
                if frame is None or frame.empty:
                    continue
                worksheet.write(row, 0, section_title, formats["section"])
                row = _write_table(worksheet, frame, row + 1, formats)

            worksheet.freeze_panes(4, 1)

    return str(path)
