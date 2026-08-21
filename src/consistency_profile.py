"""Measure how concentrated each strategy's daily profit is.

The 40% rule compares the best single day against total profit at the moment a
reward is requested. That ratio is scale-invariant: halving risk halves both
numbers and changes nothing. Only the *shape* of the daily P&L distribution
matters -- how many positive days there are, how similar they are, and how much
the losing days claw back.

For each strategy this runs the funded stage across the whole history, then
asks: over every rolling window of N days that ended in profit, what share did
the best day carry? That is exactly the test the firm applies.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market, Rules, run_stage  # noqa: E402
from run import load, slice_period  # noqa: E402


def daily_pnl(strategy, mkt, rules, i0, i1) -> pd.Series:
    """Realised P&L per calendar day over a long funded-mode run."""
    strategy.prepare(mkt)
    st = run_stage(mkt, strategy, rules, i0, i1, "funded", rules.initial_balance,
                   None, rules.initial_balance)
    rows = {}
    for t in st.trades:
        d = pd.Timestamp(t.ts_out).normalize()
        rows[d] = rows.get(d, 0.0) + t.pnl
    if not rows:
        return pd.Series(dtype=float)
    s = pd.Series(rows).sort_index()
    return s


def window_shares(s: pd.Series, days: int) -> dict:
    """Best-day share of total profit over each rolling window that profited."""
    if s.empty:
        return {}
    full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="D"),
                     fill_value=0.0)
    vals = full.values
    n = len(vals)
    shares, totals = [], []
    for a in range(0, n - days):
        w = vals[a:a + days]
        tot = w.sum()
        if tot <= 0:
            continue
        best = w.max()
        if best <= 0:
            continue
        shares.append(best / tot)
        totals.append(tot)
    if not shares:
        return {}
    arr = np.array(shares)
    return {
        "windows": len(arr),
        "share_med": float(np.median(arr)),
        "share_p25": float(np.percentile(arr, 25)),
        "pass_rate": float((arr <= 0.40).mean()),
        "profit_med": float(np.median(totals)),
    }


def profile(name, strategy, mkt, rules, i0, i1, horizons=(30, 60, 90)):
    s = daily_pnl(strategy, mkt, rules, i0, i1)
    if s.empty:
        return None
    out = {"strategy": name, "pos_days": int((s > 0).sum()),
           "neg_days": int((s < 0).sum()),
           "trading_days": len(s)}
    for h in horizons:
        r = window_shares(s, h)
        out[f"pass{h}"] = r.get("pass_rate", 0.0) * 100
        out[f"share{h}"] = r.get("share_med", float("nan"))
        out[f"profit{h}"] = r.get("profit_med", float("nan"))
    return out


if __name__ == "__main__":
    import strategies_final as F

    df = load("xauusd_m1_clean")
    mkt = Market(df)
    rules = Rules()
    i0, i1 = slice_period(df, "2025-02-01", None)

    rows = []
    for cls in F.FINAL:
        r = profile(cls.name.split()[0], cls(), mkt, rules, i0, i1)
        if r:
            rows.append(r)
    out = pd.DataFrame(rows)
    print("Daily-profit concentration, whole history, funded mode\n")
    print("pass30/60/90 = % of profitable rolling windows clearing the 40% rule")
    print("share30/60/90 = median best-day share over those windows\n")
    show = out.copy()
    for c in show.columns:
        if c.startswith("share"):
            show[c] = (show[c] * 100).round(0)
        elif c.startswith(("pass", "profit")):
            show[c] = show[c].round(0)
    print(show.to_string(index=False))
