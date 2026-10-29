"""Strip plots of arrow positions, one panel per layer, plus a per-layer separation curve."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette (dataviz skill), fixed order, validated light-mode.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "text.color": INK, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
})


def strips(scores: np.ndarray, cells: Sequence[tuple[str, np.ndarray]], held_out: np.ndarray,
           layers: Sequence[int], panel_layers: Sequence[int], titles: dict[int, str],
           suptitle: str, out: str | Path) -> None:
    """scores [n, layers]; cells: (name, mask over n) top to bottom; one panel per panel_layer."""
    cols = 5 if len(panel_layers) > 6 else len(panel_layers)
    rows = -(-len(panel_layers) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 0.5 * len(cells) * rows + 1.4 * rows),
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.28, 0.28, size=scores.shape[0])
    for ax, layer in zip(axes.flat, panel_layers):
        li = list(layers).index(layer)
        ax.axvline(0.5, color=AXIS, lw=1, zorder=0)
        for x in (0, 1):
            ax.axvline(x, color=GRID, lw=0.8, zorder=0)
        for row, (name, mask) in enumerate(cells):
            y = len(cells) - 1 - row
            color = SERIES[row % len(SERIES)]
            for ho, face in ((False, color), (True, "none")):
                m = mask & (held_out == ho)
                ax.scatter(scores[m, li], y + jitter[m], s=22, facecolors=face, edgecolors=color,
                           linewidths=1.2, zorder=2 if ho else 1)
        ax.set_title(titles.get(layer, f"layer {layer}"), fontsize=9, color=INK2, loc="left")
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([f"{n} ({int(m.sum())})" for n, m in reversed(cells)])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        if np.isnan(scores[:, li]).all():
            ax.set_xlim(-0.5, 1.5)
            ax.text(0.5, 0.5, "zero arrow", transform=ax.transAxes, ha="center", color=MUTED)
            continue
        lo, hi = np.nanpercentile(scores[:, li], [1, 99])
        pad = 0.1 * (hi - lo)
        ax.set_xlim(min(lo, -0.5) - pad, max(hi, 1.5) + pad)
    for ax in list(axes.flat)[len(panel_layers):]:
        ax.set_visible(False)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.005, "position on the arrow: 0 = training negative mean, 1 = training positive mean, "
             "line at 0.5 = threshold.  filled = training item, hollow = held out.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def curves(layers: Sequence[int], series: Sequence[tuple[str, Sequence[float]]], band: tuple[Sequence[float], Sequence[float]] | None,
           ylabel: str, title: str, out: str | Path, chance: float = 0.5) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.axhline(chance, color=AXIS, lw=1)
    if band is not None:
        ax.fill_between(layers, band[0], band[1], color=GRID, lw=0, label="shuffled-label AUC, 5th–95th pct")
    for i, (name, ys) in enumerate(series):
        ys = np.array([np.nan if y is None else y for y in ys], dtype=float)
        ax.plot(layers, ys, color=SERIES[i], lw=2, marker="o", ms=4, label=name)
    ax.set_xlabel("layer")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
