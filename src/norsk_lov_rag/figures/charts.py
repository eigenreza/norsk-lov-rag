"""Two chart forms used by most figures: rows of dots with intervals, and horizontal bars."""

from __future__ import annotations

import math

from matplotlib.ticker import PercentFormatter

from norsk_lov_rag.figures.style import INK, MARKERS, PALETTE


def _value(entry) -> float | None:
    if entry is None:
        return None
    if isinstance(entry, dict):
        return entry.get("mean")
    return float(entry)


def dot_rows(
    ax,
    rows: list[str],
    series: list[tuple[str, list]],
    xlabel: str,
    percent: bool = True,
    full_range: bool = False,
) -> None:
    """One row per label, one dot per series, with the bootstrap interval as a thin line.

    Values are dicts with mean, low and high, as the result files store them, or None where
    a value is missing.
    """
    count = len(series)
    step = min(0.22, 0.7 / max(count, 1))
    lows = []
    for i, (label, values) in enumerate(series):
        offset = (i - (count - 1) / 2) * step
        xs, ys, below, above = [], [], [], []
        for row, entry in enumerate(values):
            mean = _value(entry)
            if mean is None:
                continue
            low = entry.get("low") if isinstance(entry, dict) else None
            high = entry.get("high") if isinstance(entry, dict) else None
            xs.append(mean)
            ys.append(row + offset)
            below.append(max(0.0, mean - low) if low is not None else 0.0)
            above.append(max(0.0, high - mean) if high is not None else 0.0)
            lows.append(low if low is not None else mean)
        if not xs:
            continue
        ax.errorbar(
            xs, ys, xerr=[below, above], fmt=MARKERS[i % len(MARKERS)],
            color=PALETTE[i % len(PALETTE)], ms=5.5, elinewidth=1.2, capsize=0,
            markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3,
        )  # fmt: skip
    ax.set_yticks(range(len(rows)), rows)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    if percent:
        start = 0.0
        if not full_range and lows:
            start = max(0.0, math.floor((min(lows) - 0.03) * 10) / 10)
        ax.set_xlim(start, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def bars(
    ax,
    rows: list[str],
    values: list[float],
    xlabel: str,
    percent: bool = False,
    color: str = PALETTE[0],
    fmt: str = "{:,.0f}",
    sep: str = " ",
) -> None:
    """Horizontal bars from zero, with the value at the end of each bar.

    ``sep`` separates thousands: a space in Norwegian, a comma in English.
    """
    ax.barh(range(len(rows)), values, height=0.6, color=color, zorder=2)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    top = max(values) if values else 1.0
    for row, value in enumerate(values):
        text = f"{value:.0%}" if percent else fmt.format(value).replace(",", sep)
        ax.text(value + top * 0.01, row, text, va="center", ha="left", fontsize=8, color=INK)
    ax.set_xlim(0, top * 1.15 if top > 0 else 1)
    if percent:
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def legend_below(fig, axes, ncol: int | None = None) -> None:
    """One legend under the figure, taken from the first axes that has labelled series."""
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                ncol=ncol or min(len(labels), 4),
            )  # fmt: skip
            return
