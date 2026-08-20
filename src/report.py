"""Reporting: equity curves, stage tables and cross-strategy correlation."""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "reports")

BG = "#0d1117"
FG = "#c9d1d9"
GRID = "#21262d"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
AMBER = "#d29922"
PURPLE = "#bc8cff"


def _style(ax):
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    ax.yaxis.label.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.title.set_color(FG)


def equity_series(stage, mkt) -> pd.DataFrame:
    """Balance samples for one stage as a time-indexed frame."""
    if not stage.equity_idx:
        return pd.DataFrame(columns=["ts", "equity"])
    ts = mkt.ts[np.array(stage.equity_idx)]
    return pd.DataFrame({"ts": ts, "equity": np.array(stage.equity_val)})


def journey_frame(stages, mkt, initial: float) -> pd.DataFrame:
    """Stitch every stage into one continuous record.

    `equity` is the live account balance, which drops back to base on each
    payout. `cumulative` adds withdrawn profit back so the line shows true
    performance rather than the sawtooth of withdrawals.
    """
    parts = []
    withdrawn_before = 0.0
    for st in stages:
        d = equity_series(st, mkt)
        if d.empty:
            continue
        d = d.copy()
        d["stage"] = st.name
        # withdrawals that had already happened by each sample
        pay_ts = [p["ts"] for p in st.payouts]
        pay_amt = np.cumsum([p["gross"] for p in st.payouts]) if st.payouts else []
        taken = np.zeros(len(d))
        for k, t in enumerate(pay_ts):
            taken[d["ts"].values > t] = pay_amt[k]
        d["cumulative"] = d["equity"] + taken + withdrawn_before
        if st.payouts:
            withdrawn_before += float(np.sum([p["gross"] for p in st.payouts]))
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["ts", "equity", "cumulative", "stage"])
    return pd.concat(parts, ignore_index=True).sort_values("ts").reset_index(drop=True)


def plot_journey(name, stages, mkt, rules, path):
    """Full challenge -> funded equity curve, with rule limits drawn in."""
    jf = journey_frame(stages, mkt, rules.initial_balance)
    if jf.empty:
        return None
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor(BG)
    _style(ax)
    _style(ax2)

    colors = {"phase1": ACCENT, "phase2": PURPLE, "funded": GREEN}
    for st in stages:
        seg = jf[jf["stage"] == st.name]
        if seg.empty:
            continue
        ax.plot(seg["ts"], seg["cumulative"], color=colors[st.name], linewidth=1.5,
                label=f"{st.name}")
        ax.axvspan(seg["ts"].iloc[0], seg["ts"].iloc[-1], color=colors[st.name], alpha=0.05)

    init = rules.initial_balance
    ax.axhline(init, color=FG, linewidth=0.9, alpha=0.5)
    ax.axhline(init * (1 - rules.max_loss), color=RED, linestyle="--", linewidth=1.2,
               label=f"max loss floor (${init * (1 - rules.max_loss):,.0f})")
    ax.axhline(init * (1 + rules.p1_target), color=ACCENT, linestyle=":", linewidth=1.2,
               label=f"P1 target (+{rules.p1_target:.0%})")
    ax.axhline(init * (1 + rules.p2_target), color=PURPLE, linestyle=":", linewidth=1.2,
               label=f"P2 target (+{rules.p2_target:.0%})")

    # mark stage transitions and payouts
    for st in stages:
        if st.passed:
            ax.axvline(mkt.ts[st.end_idx], color=GREEN, linewidth=1.0, alpha=0.6)
        if st.breached:
            ax.axvline(mkt.ts[st.end_idx], color=RED, linewidth=1.4)
        for p in st.payouts:
            ax.scatter([p["ts"]], [p["balance_before"] + 0], color=AMBER, s=28, zorder=5,
                       marker="v")

    ax.set_ylabel("equity, profit re-added (USD)")
    ax.set_title(f"{name} - Stellar 2-Step $6K journey", fontsize=12, pad=12)
    leg = ax.legend(loc="upper left", fontsize=8, facecolor=BG, edgecolor=GRID, ncol=2)
    for t in leg.get_texts():
        t.set_color(FG)

    # drawdown panel
    peak = jf["cumulative"].cummax()
    dd = (jf["cumulative"] - peak) / peak * 100
    ax2.fill_between(jf["ts"], dd, 0, color=RED, alpha=0.45)
    ax2.plot(jf["ts"], dd, color=RED, linewidth=0.8)
    ax2.axhline(-rules.max_loss * 100, color=RED, linestyle="--", linewidth=1.0)
    ax2.set_ylabel("drawdown %")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
    return path


def plot_funded_only(name, stages, mkt, rules, path):
    """Post-pass performance: what the account did after it got funded."""
    funded = [s for s in stages if s.name == "funded"]
    if not funded:
        return None
    st = funded[0]
    jf = journey_frame([st], mkt, rules.initial_balance)
    if jf.empty:
        return None

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    _style(ax)
    init = rules.initial_balance
    ax.plot(jf["ts"], jf["cumulative"], color=GREEN, linewidth=1.8,
            label="cumulative (withdrawals added back)")
    ax.plot(jf["ts"], jf["equity"], color=ACCENT, linewidth=1.0, alpha=0.75,
            label="account balance")
    ax.axhline(init, color=FG, linewidth=0.9, alpha=0.5)
    ax.axhline(init * (1 - rules.max_loss), color=RED, linestyle="--", linewidth=1.2,
               label="max loss floor")

    total = 0.0
    for p in st.payouts:
        total += p["net"]
        ax.axvline(p["ts"], color=AMBER, linewidth=1.0, alpha=0.7)
        ax.annotate(f"payout ${p['net']:,.0f}", xy=(p["ts"], p["balance_before"]),
                    xytext=(0, 12), textcoords="offset points", color=AMBER, fontsize=8,
                    ha="center")

    ax.set_title(
        f"{name} - funded account performance after passing "
        f"({len(st.payouts)} payouts, ${total:,.0f} net to trader)",
        fontsize=12, pad=12,
    )
    ax.set_ylabel("USD")
    leg = ax.legend(loc="upper left", fontsize=9, facecolor=BG, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(FG)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
    return path


def plot_portfolio(results, mkt, rules, path):
    """All four accounts on one axis, plus their combined equity."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor(BG)
    _style(ax)
    palette = [ACCENT, PURPLE, GREEN, AMBER]
    for (name, stages), col in zip(results, palette):
        jf = journey_frame(stages, mkt, rules.initial_balance)
        if jf.empty:
            continue
        ax.plot(jf["ts"], jf["cumulative"] - rules.initial_balance, color=col,
                linewidth=1.3, label=name, alpha=0.9)
    ax.axhline(0, color=FG, linewidth=0.9, alpha=0.5)
    ax.set_ylabel("net profit per account (USD)")
    ax.set_title("All four accounts, Dec 2025 - Aug 2026", fontsize=12, pad=12)
    leg = ax.legend(loc="upper left", fontsize=8, facecolor=BG, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(FG)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)
    return path


def daily_pnl(stages, mkt) -> pd.Series:
    """Realised P&L per calendar day across all stages, for correlation work."""
    rows = []
    for st in stages:
        for t in st.trades:
            rows.append((pd.Timestamp(t.ts_out).normalize(), t.pnl))
    if not rows:
        return pd.Series(dtype=float)
    d = pd.DataFrame(rows, columns=["day", "pnl"])
    return d.groupby("day")["pnl"].sum()


def correlation_table(results) -> pd.DataFrame:
    series = {}
    for name, stages in results:
        s = daily_pnl(stages, None)
        if not s.empty:
            series[name] = s
    if len(series) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(series).fillna(0.0)
    return df.corr()
