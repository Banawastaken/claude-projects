"""Chart: strategy expectancy per instrument, both windows side by side.

An instrument only earns a recommendation if it is positive in BOTH windows,
so the chart encodes that directly rather than ranking on an average that a
single lucky window could carry.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report import BG, FG, GRID, _style  # noqa: E402

GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
BLUE = "#58a6ff"

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")


def build():
    d = pd.read_csv(os.path.join(REPORTS, "instrument_edge.csv"))
    d = d.sort_values("exp_avg", ascending=True)
    both_pos = (d["exp_DEV"] > 0) & (d["exp_TEST"] > 0)

    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 10))
    fig.patch.set_facecolor(BG)
    _style(ax)

    h = 0.38
    ax.barh(y + h / 2, d["exp_DEV"], height=h, color=BLUE, alpha=0.85,
            label="2025 (development)")
    ax.barh(y - h / 2, d["exp_TEST"], height=h, color=AMBER, alpha=0.85,
            label="Dec 2025 - Aug 2026 (test)")

    ax.axvline(0, color=FG, linewidth=1.0, alpha=0.6)
    ax.set_yticks(y)
    labels = []
    for name, ok in zip(d["instrument"], both_pos):
        labels.append(f"{'* ' if ok else '  '}{name}")
    ax.set_yticklabels(labels, fontsize=9)
    for tick, ok in zip(ax.get_yticklabels(), both_pos):
        tick.set_color(GREEN if ok else FG)
        if ok:
            tick.set_fontweight("bold")

    ax.set_xlabel("expectancy per trade (R)")
    ax.set_title("A3 Donchian H4 on every FundedNext CFD\n"
                 "starred instruments are profitable in BOTH windows",
                 fontsize=12, pad=14)
    leg = ax.legend(loc="lower right", fontsize=9, facecolor=BG, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(FG)
    ax.set_ylim(-1, len(d))
    fig.tight_layout()
    out = os.path.join(REPORTS, "instrument_edge.png")
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", build())
