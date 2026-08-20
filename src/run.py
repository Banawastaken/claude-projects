"""Run strategies through the full Stellar 2-Step $6K journey and report."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market, Rules, run_challenge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def load(tag: str = "xauusd_m1") -> pd.DataFrame:
    df = pd.read_parquet(os.path.join(DATA, f"{tag}.parquet"))
    return df.sort_values("ts").reset_index(drop=True)


def slice_period(df: pd.DataFrame, start: str, end: str | None = None):
    m = df["ts"] >= pd.Timestamp(start, tz="UTC")
    if end:
        m &= df["ts"] < pd.Timestamp(end, tz="UTC")
    idx = np.flatnonzero(m.values)
    return int(idx[0]), int(idx[-1]) + 1


def trade_stats(trades):
    if not trades:
        return {}
    pnl = np.array([t.pnl for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gp, gl = wins.sum(), -losses.sum()
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(pnl),
        "avg_win": wins.mean() if len(wins) else 0.0,
        "avg_loss": losses.mean() if len(losses) else 0.0,
        "profit_factor": (gp / gl) if gl > 0 else float("inf"),
        "net": pnl.sum(),
        "best": pnl.max(),
        "worst": pnl.min(),
    }


def summarise(name, stages, rules):
    out = {"strategy": name}
    labels = {"phase1": "P1", "phase2": "P2", "funded": "FUND"}
    passed_p1 = passed_p2 = False
    for st in stages:
        p = labels[st.name]
        s = trade_stats(st.trades)
        out[f"{p}_result"] = (
            "PASS" if st.passed else ("BREACH: " + st.breach_reason if st.breached else "timeout")
        )
        out[f"{p}_days"] = round(st.calendar_days, 1)
        out[f"{p}_tdays"] = st.trading_days
        out[f"{p}_trades"] = s.get("trades", 0)
        out[f"{p}_wr"] = round(s.get("win_rate", 0) * 100, 1)
        out[f"{p}_pf"] = round(s.get("profit_factor", 0), 2)
        out[f"{p}_net"] = round(st.final_balance - rules.initial_balance, 2)
        out[f"{p}_maxdd"] = round(st.max_dd_pct * 100, 2)
        out[f"{p}_worstday"] = round(st.worst_daily_dd_pct * 100, 2)
        if st.name == "phase1":
            passed_p1 = st.passed
        if st.name == "phase2":
            passed_p2 = st.passed
        if st.name == "funded":
            out["payouts"] = len(st.payouts)
            out["payout_total"] = round(sum(p["net"] for p in st.payouts), 2)
            out["funded_alive"] = not st.breached
    out["funded"] = passed_p1 and passed_p2
    return out


def run_one(strategy, df, start, end, rules=None, verbose=True):
    rules = rules or Rules()
    mkt = Market(df)
    i0, i1 = slice_period(df, start, end)
    stages = run_challenge(mkt, strategy, rules, i0, i1)
    if verbose:
        print(f"\n=== {strategy.name}")
        for st in stages:
            verdict = "PASS" if st.passed else ("BREACH " + st.breach_reason if st.breached else "no result")
            print(
                f"  {st.name:7s} {verdict:35s} bal={st.final_balance:8.2f} "
                f"trades={len(st.trades):4d} days={st.calendar_days:6.1f} "
                f"maxDD={st.max_dd_pct * 100:5.2f}% worstDay={st.worst_daily_dd_pct * 100:5.2f}%"
            )
            if st.payouts:
                print(f"          payouts: {len(st.payouts)} totalling ${sum(p['net'] for p in st.payouts):,.2f}")
    return stages, mkt
