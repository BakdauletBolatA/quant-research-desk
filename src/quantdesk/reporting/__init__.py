"""Charts, the HTML research report and the Excel deliverable."""

from __future__ import annotations

from quantdesk.reporting import charts
from quantdesk.reporting.excel import write_workbook
from quantdesk.reporting.style import apply_style
from quantdesk.reporting.tearsheet import (
    FigureCounter,
    bullets,
    callout,
    frame_to_html,
    heading,
    render_tearsheet,
    table,
    text,
)

__all__ = [
    "FigureCounter",
    "apply_style",
    "bullets",
    "callout",
    "charts",
    "frame_to_html",
    "heading",
    "render_tearsheet",
    "table",
    "text",
    "write_workbook",
]
