"""Chart: measured edge against the edge a challenge actually requires."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report import BG, FG, GRID, _style  # noqa: E402

GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")

# concept, gross expR, net expR, n trades, t-stat
DATA = [
    ("Turn of month", 0.055, 0.043, 641, 1.73),
    ("Time-series momentum", 0.062, 0.010, 2317, 1.93),
    ("Short-term reversal", 0.038, 0.004, 2247, 1.38),
    ("A3 Donchian (control)", 0.046, -0.034, 4267, 1.87),
    ("Volatility contraction", -0.031, -0.078, 1505, -0.90),
]
NEEDED = 0.30


def build():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    fig.patch.set_facecolor(BG)
    _style(ax)

    names = [d[0] for d in DATA]
    gross = np.array([d[1] for d in DATA])
    net = np.array([d[2] for d in DATA])
    y = np.arange(len(DATA))
    h = 0.36

    ax.barh(y + h / 2, gross, height=h, color=BLUE, alpha=0.9,
            label="gross, before any costs")
    ax.barh(y - h / 2, net, height=h,
            color=[GREEN if v > 0 else RED for v in net], alpha=0.9,
            label="net, after spread, commission and slippage")

    ax.axvline(0, color=FG, linewidth=1.0, alpha=0.7)
    ax.axvline(NEEDED, color=AMBER, linewidth=2.0, linestyle="--")
    ax.annotate("what a 2-step challenge needs\n(+0.30 R per trade)",
                xy=(NEEDED, len(DATA) - 0.5), xytext=(NEEDED + 0.012, len(DATA) - 0.75),
                color=AMBER, fontsize=10, fontweight="bold", va="top")

    for i, (_, g, n, cnt, t) in enumerate(DATA):
        ax.annotate(f"t={t:+.2f}  n={cnt:,}", xy=(max(g, 0.0) + 0.006, i + h / 2),
                    va="center", color=FG, fontsize=8, alpha=0.75)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("expectancy per trade (R)")
    ax.set_xlim(-0.10, NEEDED + 0.10)
    ax.set_title("Every concept tested, against the bar that matters\n"
                 "9 instruments, 2015-2020 design window",
                 fontsize=12, pad=14)
    leg = ax.legend(loc="lower right", fontsize=9, facecolor=BG, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.tight_layout()
    out = os.path.join(REPORTS, "edge_gap.png")
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", build())
