"""Chart: A3's expectancy year by year, 2015-2026."""

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

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")


def build():
    d = pd.read_csv(os.path.join(REPORTS, "decade_edge.csv"))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(BG)
    _style(ax)
    _style(ax2)

    built_in = d["year"].isin([2025, 2026])
    colors = [AMBER if b else (GREEN if v > 0 else RED)
              for b, v in zip(built_in, d["expectancy_R"])]
    ax.bar(d["year"], d["expectancy_R"], color=colors, width=0.68)
    ax.axhline(0, color=FG, linewidth=1.0, alpha=0.7)
    ax.set_ylabel("expectancy per trade (R)")
    ax.set_title("A3 Donchian H4 on gold, year by year\n"
                 "amber = the window the strategy was built and tested in",
                 fontsize=12, pad=14)

    pre = d[d["year"] <= 2024]
    ax.annotate(f"2015-2024: {pre['total_R'].sum():+.0f} R in total",
                xy=(0.02, 0.06), xycoords="axes fraction", color=RED,
                fontsize=10, fontweight="bold")

    cum = d["total_R"].cumsum()
    ax2.plot(d["year"], cum, color=FG, linewidth=1.8, marker="o", markersize=4)
    ax2.axhline(0, color=FG, linewidth=1.0, alpha=0.5)
    ax2.fill_between(d["year"], cum, 0, where=(cum >= 0), color=GREEN, alpha=0.25)
    ax2.fill_between(d["year"], cum, 0, where=(cum < 0), color=RED, alpha=0.25)
    ax2.set_ylabel("cumulative R")
    ax2.set_xlabel("year")
    ax2.set_xticks(d["year"])

    fig.tight_layout()
    out = os.path.join(REPORTS, "decade_edge.png")
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", build())
