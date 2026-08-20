"""Diagnose when and why the 2025 edge broke down in 2026.

Reinventing a strategy without understanding the failure just produces a
different strategy fitted to a different accident. This measures each
strategy's raw expectancy month by month across the whole span, alongside a
regime description of gold in that month, so the fix can address the actual
failure mode.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market  # noqa: E402
from evaluate import unlimited_rules  # noqa: E402
from engine import run_stage  # noqa: E402
from indicators import adx, atr, resample  # noqa: E402
from run import load, slice_period  # noqa: E402

import strategies_final as F  # noqa: E402


def monthly_r(strategy, mkt, i0, i1):
    rules = unlimited_rules()
    strategy.prepare(mkt)
    st = run_stage(mkt, strategy, rules, i0, i1, "funded", rules.initial_balance, None,
                   rules.initial_balance)
    rows = []
    for t in st.trades:
        if t.risk_usd > 0:
            rows.append((pd.Timestamp(t.ts_out).to_period("M"), t.pnl / t.risk_usd))
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows, columns=["month", "R"])
    return d.groupby("month")["R"].agg(["count", "sum", "mean"])


def regime_table(mkt, df):
    """Per-month gold regime: realised trend strength vs chop."""
    d1 = resample(mkt, 1440)
    a = atr(d1, 14)
    ax = adx(d1, 14)
    ts = mkt.ts[d1.start_idx]
    months = pd.Series(pd.to_datetime(ts)).dt.to_period("M")
    close = d1.c
    out = []
    for m in months.unique():
        sel = (months == m).values
        if sel.sum() < 5:
            continue
        c = close[sel]
        net = abs(c[-1] - c[0])
        path = np.abs(np.diff(c)).sum()
        out.append({
            "month": m,
            "ret_pct": (c[-1] / c[0] - 1) * 100,
            "efficiency": net / path if path > 0 else 0.0,  # 1 = pure trend, 0 = pure chop
            "adx": np.nanmean(ax[sel]),
            "atr_pct": np.nanmean(a[sel]) / np.nanmean(c) * 100,
        })
    return pd.DataFrame(out).set_index("month")


if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    mkt = Market(df)
    i0, i1 = slice_period(df, "2025-02-01", None)

    reg = regime_table(mkt, df)
    print("Gold regime by month (efficiency: 1.0 = clean trend, 0 = pure chop)\n")
    print(reg.round(2).to_string())

    print("\n\nMonthly expectancy in R per strategy (limits off)\n")
    frames = {}
    for cls in F.FINAL:
        m = monthly_r(cls(), mkt, i0, i1)
        if m.empty:
            continue
        frames[cls.name.split()[0]] = m["mean"]
    tbl = pd.DataFrame(frames)
    tbl["n_pos"] = (tbl > 0).sum(axis=1)
    print(tbl.round(2).to_string())

    dev = tbl.loc[tbl.index < pd.Period("2025-12", "M")]
    test = tbl.loc[tbl.index >= pd.Period("2025-12", "M")]
    print("\nmean expectancy R, dev vs test:")
    cols = [c for c in tbl.columns if c != "n_pos"]
    print(pd.DataFrame({
        "dev (2025)": dev[cols].mean().round(3),
        "test (Dec25-Aug26)": test[cols].mean().round(3),
    }).to_string())
    print("\nregime means:")
    print(pd.DataFrame({
        "dev": reg.loc[reg.index < pd.Period("2025-12", "M")].mean().round(2),
        "test": reg.loc[reg.index >= pd.Period("2025-12", "M")].mean().round(2),
    }).to_string())
