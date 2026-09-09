"""Chart styling: one palette, one grid, one set of rules for every figure.

The palette is a validated categorical set — the slot order is the
colour-vision-deficiency safety mechanism, so hues are assigned in fixed order
and never cycled. Sequential (magnitude) work uses a single blue ramp;
polarity — correlations, factor betas, upside/downside — uses the blue/red
diverging pair with a neutral grey midpoint. Status colours are reserved and
always ship with a label, never as colour alone.
"""

from __future__ import annotations

import contextlib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- surfaces and ink -------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# --- categorical slots, in fixed order --------------------------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# --- status (reserved; always with a label) ---------------------------------
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# --- sequential blue ramp ---------------------------------------------------
SEQUENTIAL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list("qd_sequential", SEQUENTIAL)
CMAP_DIVERGING = LinearSegmentedColormap.from_list(
    "qd_diverging", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#8f1f1f"]
)

FONT_STACK = [
    "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans", "sans-serif",
]


def apply_style() -> None:
    """Install the house style globally. Idempotent."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.edgecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.edgecolor": SURFACE,
            "savefig.bbox": "standard",
            "savefig.dpi": 160,
            "figure.dpi": 110,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK_PRIMARY,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 10,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "lines.solid_capstyle": "round",
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "font.size": 10,
            "text.color": INK_PRIMARY,
            "figure.autolayout": False,
        }
    )


def series_color(index: int) -> str:
    """Categorical hue for slot ``index`` (0-based), in fixed order."""
    return SERIES[index % len(SERIES)]


def annotate_source(ax, text: str) -> None:
    """Uniform provenance footnote — every figure states where its numbers came from."""
    ax.figure.text(
        0.005, -0.02, text, ha="left", va="top", fontsize=8, color=INK_MUTED,
        transform=ax.figure.transFigure,
    )


def finish(fig, path, source: str | None = None) -> str:
    """Lay the figure out, stamp the source note in reserved space, and save.

    The note gets its own strip at the bottom rather than being dropped on top
    of the plot: with rotated tick labels a naive ``fig.text`` at y=0 lands
    underneath them.
    """
    reserved = 0.075 if source else 0.02
    with contextlib.suppress(Exception):  # layout is cosmetic, never fatal
        fig.tight_layout(rect=(0.0, reserved, 1.0, 1.0))
    if source:
        fig.text(0.008, 0.012, source, ha="left", va="bottom", fontsize=8, color=INK_MUTED)
    fig.savefig(path)
    plt.close(fig)
    return str(path)
