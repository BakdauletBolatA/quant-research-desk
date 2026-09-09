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
    "charts",
    "apply_style",
    "write_workbook",
    "render_tearsheet",
    "frame_to_html",
    "FigureCounter",
    "table",
    "text",
    "heading",
    "callout",
    "bullets",
]
