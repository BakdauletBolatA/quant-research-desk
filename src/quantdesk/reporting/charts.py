"""Every figure in the report, one function each.

Rules applied uniformly:

* Categorical hues are assigned in fixed slot order and never cycled. Where a
  chart would need more than eight identities (16 tickers, 9 sectors) the form
  changes instead of the palette — a magnitude heatmap or a single-hue bar
  chart, not nine invented colours.
* Two to four series get a legend *and* direct labels, so identity never rests
  on colour alone. Five or more get a legend.
* Magnitude uses the single-hue blue ramp; polarity (correlations, factor
  betas, upside) uses the blue/red diverging pair with a neutral midpoint.
* Every figure carries a source note.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter

from quantdesk.reporting.style import (
    BASELINE,
    CMAP_DIVERGING,
    CMAP_SEQUENTIAL,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES,
    STATUS,
    SURFACE,
    apply_style,
    finish,
    series_color,
)

apply_style()

SOURCE_MARKET = "Source: Yahoo Finance adjusted closes · QuantDesk"
SOURCE_FACTORS = "Source: Kenneth R. French Data Library · QuantDesk"
SOURCE_MODEL = "Source: QuantDesk model output · analyst input sheet"

_pct = PercentFormatter(xmax=1.0, decimals=0)


def _direct_label(ax, x, y, text: str, color: str, dx: float = 6.0) -> None:
    ax.annotate(
        text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
        va="center", ha="left", fontsize=9, color=color, fontweight="bold",
        clip_on=False,
    )


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def equity_curves(
    returns: pd.DataFrame, path: Path, benchmark: pd.Series | None = None, title: str | None = None
) -> str:
    """Compounded growth of 1 unit, log scale (equal vertical distance = equal return)."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    curves = (1.0 + returns).cumprod()

    for i, column in enumerate(curves.columns):
        ax.plot(curves.index, curves[column], color=series_color(i), label=str(column), zorder=3)

    if benchmark is not None:
        bench_curve = (1.0 + benchmark.reindex(returns.index).fillna(0.0)).cumprod()
        ax.plot(bench_curve.index, bench_curve, color=INK_MUTED, linewidth=1.8,
                linestyle=(0, (5, 3)), label=str(benchmark.name or "Benchmark"), zorder=2)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}x"))
    ax.set_title(title or "Growth of 1.00 — net of transaction costs (log scale)")
    ax.set_ylabel("Cumulative growth")
    ax.legend(ncols=3, loc="upper left", borderpad=0)
    ax.margins(x=0.01)
    return finish(fig, path, SOURCE_MARKET)


def drawdown_chart(returns: pd.DataFrame, path: Path, highlight: list[str] | None = None) -> str:
    """Underwater plot. Two named series are filled and directly labelled."""
    from quantdesk.analytics.performance import drawdown_series

    fig, ax = plt.subplots(figsize=(11, 4.4))
    highlight = highlight or list(returns.columns[:2])

    for column in returns.columns:
        dd = drawdown_series(returns[column])
        if column in highlight:
            slot = highlight.index(column)
            ax.fill_between(dd.index, dd.to_numpy(), 0, color=series_color(slot), alpha=0.16,
                            zorder=2, linewidth=0)
            ax.plot(dd.index, dd, color=series_color(slot), label=str(column), zorder=4)
            _direct_label(ax, dd.index[-1], float(dd.iloc[-1]), str(column), series_color(slot))
        else:
            ax.plot(dd.index, dd, color=GRIDLINE, linewidth=1.1, zorder=1)

    ax.axhline(0, color=BASELINE, linewidth=1.0)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_title("Drawdown from running peak")
    ax.set_ylabel("Drawdown")
    ax.legend(loc="lower left", ncols=2)
    ax.margins(x=0.01)
    return finish(fig, path, SOURCE_MARKET)


def rolling_metric(
    series: dict[str, pd.Series], path: Path, title: str, ylabel: str, as_percent: bool = False
) -> str:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, (label, values) in enumerate(series.items()):
        ax.plot(values.index, values, color=series_color(i), label=label)
        if len(series) <= 4 and values.notna().any():
            last = values.dropna()
            _direct_label(ax, last.index[-1], float(last.iloc[-1]), label, series_color(i))
    if as_percent:
        ax.yaxis.set_major_formatter(_pct)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="upper left", ncols=min(len(series), 3))
    ax.margins(x=0.01)
    return finish(fig, path, SOURCE_MARKET)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------
def correlation_heatmap(corr: pd.DataFrame, path: Path, title: str | None = None) -> str:
    """Correlation is polarity data: diverging palette, neutral at zero, fixed [-1, 1]."""
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=(max(7.5, n * 0.55), max(6.4, n * 0.5)))
    image = ax.imshow(corr.to_numpy(), cmap=CMAP_DIVERGING, vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=8.5)
    ax.set_yticks(range(n), corr.index, fontsize=8.5)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if n <= 20:  # values are legible at this size, so label them
        values = corr.to_numpy()
        for i in range(n):
            for j in range(n):
                v = values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                        color="#ffffff" if abs(v) > 0.62 else INK_SECONDARY)

    bar = fig.colorbar(image, ax=ax, fraction=0.036, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_MUTED)
    ax.set_title(title or "Return correlation matrix")
    return finish(fig, path, SOURCE_MARKET)


VAR_METHOD_LABELS = {
    "historical": "Historical",
    "gaussian": "Gaussian",
    "cornish_fisher": "Cornish-Fisher",
    "ewma": "EWMA (λ=0.94)",
    "filtered_historical": "Filtered HS",
}


def var_exceptions(
    returns: pd.Series, var_forecast: pd.Series, path: Path, confidence: float, method: str
) -> str:
    """Daily returns against the rolling VaR line, with breaches called out."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    aligned = pd.concat([returns.rename("r"), var_forecast.rename("var")], axis=1).dropna()
    breaches = aligned[aligned["r"] < -aligned["var"]]

    ax.plot(aligned.index, aligned["r"], color=GRIDLINE, linewidth=0.7, zorder=1)
    ax.plot(aligned.index, -aligned["var"], color=series_color(0), linewidth=1.6, zorder=3,
            label=f"{confidence:.0%} VaR forecast — {VAR_METHOD_LABELS.get(method, method)}")
    ax.scatter(breaches.index, breaches["r"], s=26, color=STATUS["critical"], zorder=4,
               edgecolor="#ffffff", linewidth=0.6,
               label=f"Exceptions: {len(breaches)} of {len(aligned)} "
                     f"({len(breaches) / len(aligned):.2%})")

    ax.axhline(0, color=BASELINE, linewidth=0.9)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("VaR backtest — out-of-sample daily exceptions")
    ax.set_ylabel("Daily return")
    ax.legend(loc="lower left", ncols=2)
    ax.margins(x=0.01)
    return finish(fig, path, SOURCE_MARKET)


def var_model_comparison(table: pd.DataFrame, path: Path, expected_rate: float) -> str:
    """Realised exception rate per estimator against the rate the model promises."""
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    labels = [VAR_METHOD_LABELS.get(str(i), str(i)) for i in table.index]
    rates = table["exception_rate"].to_numpy(dtype=float)
    passed = table["kupiec_p"].to_numpy(dtype=float) > 0.05

    colors = [STATUS["good"] if ok else STATUS["critical"] for ok in passed]
    ax.bar(labels, rates, color=colors, width=0.58, zorder=3)
    ax.set_ylim(0, max(float(rates.max()) * 1.42, expected_rate * 2.2))

    for x, (rate, ok, p_value) in enumerate(
        zip(rates, passed, table["kupiec_p"].to_numpy(dtype=float), strict=True)
    ):
        ax.annotate(
            f"{rate:.2%}\n{'coverage OK' if ok else 'reject'} (p={p_value:.3f})",
            xy=(x, rate), xytext=(0, 7), textcoords="offset points", ha="center",
            fontsize=8.5, color=INK_SECONDARY,
        )

    ax.axhline(expected_rate, color=INK_PRIMARY, linewidth=1.4, linestyle=(0, (4, 3)), zorder=4)
    ax.annotate(
        f"the model promises {expected_rate:.1%}", xy=(0.0, expected_rate),
        xycoords=("axes fraction", "data"), xytext=(4, 5), textcoords="offset points",
        ha="left", va="bottom", fontsize=8.5, color=INK_PRIMARY, fontweight="bold",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_title("VaR model validation — realised exception rate vs promised")
    ax.set_ylabel("Exception rate")
    return finish(fig, path, SOURCE_MARKET)


def risk_contribution_chart(contributions: pd.DataFrame, path: Path) -> str:
    """Share of total portfolio risk by holding, for up to four strategies."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    assets = contributions.index.tolist()
    x = np.arange(len(assets))
    n = contributions.shape[1]
    width = 0.8 / n

    for i, column in enumerate(contributions.columns):
        ax.bar(x + i * width - 0.4 + width / 2, contributions[column], width=width * 0.9,
               color=series_color(i), label=str(column), zorder=3)

    ax.set_xticks(x, assets, rotation=0, fontsize=9)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, float(np.nanmax(contributions.to_numpy())) * 1.10)
    ax.set_title("Share of total portfolio risk by holding", pad=26)
    ax.set_ylabel("Risk contribution")
    # Legend above the plot: with a 25% concentration cap the tallest bar
    # reaches the top-right corner, where an in-axes legend would sit on it.
    ax.legend(ncols=min(n, 4), loc="lower left", bbox_to_anchor=(0.0, 1.01))
    return finish(fig, path, SOURCE_MARKET)


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def _spread_labels(positions: list[float], min_gap: float) -> list[float]:
    """Push overlapping label anchors apart while preserving their order.

    Matplotlib will happily draw two annotations on top of each other; on a
    scatter of allocator outcomes that is exactly where the interesting points
    cluster, so the labels are de-collided explicitly.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    adjusted = list(positions)
    for rank, index in enumerate(order):
        if rank == 0:
            continue
        previous = adjusted[order[rank - 1]]
        if adjusted[index] - previous < min_gap:
            adjusted[index] = previous + min_gap
    return adjusted


def efficient_frontier_chart(
    frontier: pd.DataFrame,
    assets: pd.DataFrame,
    portfolios: pd.DataFrame,
    path: Path,
) -> str:
    """Frontier, individual names, and where each allocator lands.

    Strategy points are separated by marker shape *and* direct label, not by
    colour: an eight-way colour split would breach the all-pairs CVD floor.
    """
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.plot(frontier["volatility"], frontier["expected_return"], color=series_color(0),
            linewidth=2.2, zorder=3, label="Efficient frontier")

    ax.scatter(assets["volatility"], assets["expected_return"], s=34, color=GRIDLINE,
               edgecolor=BASELINE, linewidth=0.7, zorder=2)

    all_y = list(assets["expected_return"]) + list(portfolios["expected_return"])
    span = (max(all_y) - min(all_y)) or 1.0
    ax.set_ylim(min(all_y) - span * 0.10, max(all_y) + span * 0.12)
    all_x = list(assets["volatility"]) + list(portfolios["volatility"])
    x_span = (max(all_x) - min(all_x)) or 1.0
    ax.set_xlim(min(all_x) - x_span * 0.06, max(all_x) + x_span * 0.16)

    asset_labels = _spread_labels(list(assets["expected_return"]), span * 0.032)
    for (name, row), label_y in zip(assets.iterrows(), asset_labels, strict=True):
        ax.annotate(str(name), xy=(row["volatility"], label_y), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, color=INK_MUTED, va="center")

    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    portfolio_labels = _spread_labels(list(portfolios["expected_return"]), span * 0.050)
    for i, ((name, row), label_y) in enumerate(
        zip(portfolios.iterrows(), portfolio_labels, strict=True)
    ):
        ax.scatter(row["volatility"], row["expected_return"], s=120, marker=markers[i % 8],
                   color=series_color(0), edgecolor=SURFACE, linewidth=1.1, zorder=5)
        ax.annotate(
            str(name), xy=(row["volatility"], row["expected_return"]),
            xytext=(row["volatility"] + x_span * 0.035, label_y), textcoords="data",
            fontsize=9, color=INK_PRIMARY, fontweight="bold", va="center", zorder=6,
            arrowprops={"arrowstyle": "-", "color": BASELINE, "linewidth": 0.8,
                        "shrinkA": 4, "shrinkB": 2},
        )

    ax.xaxis.set_major_formatter(_pct)
    ax.yaxis.set_major_formatter(_pct)
    ax.set_title("Efficient frontier, single names and allocator outcomes")
    ax.set_xlabel("Annualised volatility")
    ax.set_ylabel("Expected excess return (Black-Litterman posterior)")
    ax.legend(loc="lower right")
    return finish(fig, path, SOURCE_MODEL)


def weights_heatmap(weights: pd.DataFrame, path: Path, title: str | None = None) -> str:
    """Average allocation by holding and strategy — magnitude, so one hue."""
    fig, ax = plt.subplots(
        figsize=(max(8.5, weights.shape[1] * 1.4), max(6.6, weights.shape[0] * 0.45))
    )
    image = ax.imshow(weights.to_numpy(), cmap=CMAP_SEQUENTIAL, aspect="auto", vmin=0.0)

    ax.set_xticks(range(weights.shape[1]), weights.columns, rotation=28, ha="right", fontsize=9)
    ax.set_yticks(range(weights.shape[0]), weights.index, fontsize=9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    peak = float(np.nanmax(weights.to_numpy()))
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            value = weights.iat[i, j]
            if np.isfinite(value) and value > 0.001:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if value > peak * 0.55 else INK_SECONDARY)

    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02,
                       format=PercentFormatter(xmax=1.0, decimals=0))
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_MUTED)
    ax.set_title(title or "Average allocation by strategy")
    return finish(fig, path, SOURCE_MODEL)


def factor_exposures(betas: pd.Series, tstats: pd.Series, path: Path, title: str) -> str:
    """Factor loadings; sign is polarity so the diverging pair carries it, and
    statistical significance is marked with a label rather than a shade."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    values = betas.to_numpy(dtype=float)
    colors = [series_color(0) if v >= 0 else STATUS["critical"] for v in values]

    bars = ax.bar(betas.index, values, color=colors, width=0.62, zorder=3)
    for rect, value, t in zip(bars, values, tstats.reindex(betas.index).to_numpy(), strict=True):
        mark = "***" if abs(t) > 2.58 else "**" if abs(t) > 1.96 else ""
        ax.annotate(
            f"{value:+.2f}{mark}",
            xy=(rect.get_x() + rect.get_width() / 2, value),
            xytext=(0, 6 if value >= 0 else -14), textcoords="offset points",
            ha="center", fontsize=8.5, color=INK_SECONDARY,
        )

    ax.axhline(0, color=BASELINE, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel("Factor beta")
    ax.margins(y=0.2)
    legend = [
        Line2D([0], [0], color=series_color(0), linewidth=8, label="Positive loading"),
        Line2D([0], [0], color=STATUS["critical"], linewidth=8, label="Negative loading"),
        Line2D([0], [0], color=SURFACE, linewidth=0,
               label="** p<0.05   *** p<0.01 (Newey-West)"),
    ]
    ax.legend(handles=legend, loc="upper right", ncols=1)
    return finish(fig, path, SOURCE_FACTORS)


def attribution_chart(attribution: pd.Series, path: Path, title: str) -> str:
    """Annualised excess return split into factor contributions and residual alpha."""
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    data = attribution.drop(labels=["Total explained"], errors="ignore")
    values = data.to_numpy(dtype=float)
    colors = [
        STATUS["good"] if name == "Alpha" else (series_color(0) if v >= 0 else STATUS["critical"])
        for name, v in zip(data.index, values, strict=True)
    ]
    bars = ax.barh(list(data.index), values, color=colors, height=0.6, zorder=3)
    for rect, value in zip(bars, values, strict=True):
        ax.annotate(f"{value:+.2%}", xy=(value, rect.get_y() + rect.get_height() / 2),
                    xytext=(6 if value >= 0 else -6, 0), textcoords="offset points",
                    va="center", ha="left" if value >= 0 else "right", fontsize=9,
                    color=INK_SECONDARY)

    ax.axvline(0, color=BASELINE, linewidth=1.0)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Annualised contribution to excess return")
    ax.margins(x=0.18)
    return finish(fig, path, SOURCE_FACTORS)


# ---------------------------------------------------------------------------
# Valuation
# ---------------------------------------------------------------------------
def monte_carlo_distribution(
    values: np.ndarray, path: Path, ticker: str, price: float, base_value: float
) -> str:
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.hist(values, bins=70, color=SERIES[0], alpha=0.75, zorder=3, edgecolor=SURFACE,
            linewidth=0.4)

    p5, p95 = np.percentile(values, [5, 95])
    ax.axvspan(p5, p95, color=SERIES[0], alpha=0.08, zorder=1)
    for value, color, label in (
        (base_value, INK_PRIMARY, f"Base case {base_value:,.0f}"),
        (price, STATUS["critical"], f"Market price {price:,.0f}"),
    ):
        ax.axvline(value, color=color, linewidth=2.0, zorder=5)
        ax.annotate(label, xy=(value, ax.get_ylim()[1] * 0.94), xytext=(6, 0),
                    textcoords="offset points", fontsize=9, color=color, fontweight="bold")

    probability = float(np.mean(values > price))
    ax.set_title(
        f"{ticker} — simulated intrinsic value (20,000 paths). "
        f"P(value > price) = {probability:.1%}"
    )
    ax.set_xlabel("Intrinsic value per share")
    ax.set_ylabel("Simulated paths")
    ax.set_yticks([])
    return finish(fig, path, SOURCE_MODEL)


def sensitivity_heatmap(grid: pd.DataFrame, path: Path, ticker: str, price: float) -> str:
    """WACC × terminal growth. Magnitude, so a single-hue ramp; the price
    contour is drawn explicitly rather than left to the reader."""
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    values = grid.to_numpy(dtype=float)
    image = ax.imshow(values, cmap=CMAP_SEQUENTIAL, aspect="auto")

    ax.set_xticks(range(grid.shape[1]), [f"{c:.2%}" for c in grid.columns], fontsize=9)
    ax.set_yticks(range(grid.shape[0]), [f"{r:.2%}" for r in grid.index], fontsize=9)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ink colour follows the *colour ramp position*, not the value distribution:
    # a percentile threshold puts light-grey text on mid-ramp cells.
    finite = values[np.isfinite(values)]
    low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    span = (high - low) or 1.0
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            above = value > price
            ax.text(j, i, f"{value:,.0f}", ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if above else "normal",
                    color="#ffffff" if (value - low) / span > 0.55 else INK_SECONDARY)

    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8, color=INK_MUTED, labelcolor=INK_MUTED)
    ax.set_title(f"{ticker} — fair value per share (bold = above the {price:,.0f} market price)")
    ax.set_xlabel("Terminal growth")
    ax.set_ylabel("WACC")
    return finish(fig, path, SOURCE_MODEL)


def football_field_chart(field: pd.DataFrame, path: Path, ticker: str, price: float) -> str:
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    labels = list(field.index)
    y = np.arange(len(labels))

    lows = field["low"].to_numpy(dtype=float)
    highs = field["high"].to_numpy(dtype=float)
    lo, hi = float(np.nanmin(lows)), float(np.nanmax(highs))
    pad = (hi - lo) * 0.16 or max(abs(hi), 1.0) * 0.16
    ax.set_xlim(lo - pad, hi + pad)

    for i, (low, high) in enumerate(zip(lows, highs, strict=True)):
        ax.barh(y[i], high - low, left=low, height=0.52, color=series_color(0), alpha=0.85,
                zorder=3)
        ax.annotate(f"{low:,.0f}", xy=(low, y[i]), xytext=(-7, 0), textcoords="offset points",
                    va="center", ha="right", fontsize=8.5, color=INK_SECONDARY)
        ax.annotate(f"{high:,.0f}", xy=(high, y[i]), xytext=(7, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8.5, color=INK_SECONDARY)

    ax.axvline(price, color=STATUS["critical"], linewidth=2.0, zorder=5)
    ax.annotate(
        f"Market {price:,.0f}", xy=(price, 1.0), xycoords=("data", "axes fraction"),
        xytext=(7, -4), textcoords="offset points", va="top", ha="left", fontsize=9,
        color=STATUS["critical"], fontweight="bold", zorder=6,
    )

    ax.set_yticks(y, labels, fontsize=9.5)
    ax.set_ylim(len(labels) - 0.5, -0.7)
    ax.set_title(f"{ticker} — valuation range by methodology")
    ax.set_xlabel("Value per share")
    ax.grid(axis="y", visible=False)
    return finish(fig, path, SOURCE_MODEL)


def reverse_dcf_chart(table: pd.DataFrame, path: Path) -> str:
    """What the market must believe, against what the base case assumes."""
    fig, ax = plt.subplots(figsize=(10.4, 4.8))
    tickers = list(table.index)
    x = np.arange(len(tickers))

    ax.bar(x - 0.20, table["Base rev CAGR (5y)"], width=0.38, color=series_color(0),
           label="Base case 5y revenue CAGR", zorder=3)
    ax.bar(x + 0.20, table["Implied rev CAGR (5y)"], width=0.38, color=series_color(1),
           label="Implied by market price", zorder=3)

    for i, ticker in enumerate(tickers):
        implied = float(table.loc[ticker, "Implied rev CAGR (5y)"])
        ax.annotate(f"{implied:.0%}", xy=(x[i] + 0.20, implied), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_SECONDARY)

    ax.set_xticks(x, tickers, fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.axhline(0, color=BASELINE, linewidth=1.0)
    ax.set_title("Reverse DCF — growth the market is already paying for")
    ax.set_ylabel("5-year revenue CAGR")
    ax.legend(loc="upper right", ncols=2)
    return finish(fig, path, SOURCE_MODEL)


def implied_cost_of_capital_chart(table: pd.DataFrame, path: Path, risk_free: float) -> str:
    """Market-implied WACC per name against the base-case WACC and the risk-free rate."""
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    tickers = list(table.index)
    x = np.arange(len(tickers))

    ax.bar(x - 0.20, table["Base WACC"], width=0.38, color=series_color(0),
           label="Base-case WACC (CAPM)", zorder=3)
    ax.bar(x + 0.20, table["Implied WACC"], width=0.38, color=series_color(1),
           label="WACC implied by market price", zorder=3)
    ax.axhline(risk_free, color=INK_PRIMARY, linewidth=1.5, linestyle=(0, (4, 3)), zorder=5)
    ax.annotate(f"risk-free {risk_free:.1%}", xy=(len(tickers) - 0.6, risk_free), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=INK_PRIMARY, fontweight="bold")

    ax.set_xticks(x, tickers, fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Market-implied cost of capital vs the model's CAPM estimate")
    ax.set_ylabel("Discount rate")
    ax.legend(loc="upper right", ncols=2)
    return finish(fig, path, SOURCE_MODEL)


def upside_chart(rows: pd.DataFrame, path: Path) -> str:
    """Upside to fair value by name — polarity, so blue up / red down."""
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ordered = rows.sort_values("Upside")
    values = ordered["Upside"].to_numpy(dtype=float)
    colors = [series_color(0) if v >= 0 else STATUS["critical"] for v in values]

    bars = ax.barh(list(ordered.index), values, color=colors, height=0.62, zorder=3)
    for rect, value in zip(bars, values, strict=True):
        ax.annotate(f"{value:+.0%}", xy=(value, rect.get_y() + rect.get_height() / 2),
                    xytext=(6 if value >= 0 else -6, 0), textcoords="offset points",
                    va="center", ha="left" if value >= 0 else "right", fontsize=9,
                    color=INK_SECONDARY)

    ax.axvline(0, color=BASELINE, linewidth=1.1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_title("Upside to base-case intrinsic value")
    ax.set_xlabel("Fair value / market price − 1")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.16)
    return finish(fig, path, SOURCE_MODEL)
