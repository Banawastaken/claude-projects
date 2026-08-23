"""The wall test at minute resolution, which is how the levels are traded.

Daily bars are a coarse instrument for a claim about intraday levels: they say
whether a level was reached at some point, not what happened in the minutes
after it was. This uses the NQ minute bars already bought for the block-print
work, over the window where they overlap the option chains.

NQ trades at a basis to NDX -- carry until expiry, a fraction of a per cent --
so the levels are scaled by the day's own measured ratio rather than applied
raw. Getting that wrong would move every level by more than the effect being
looked for.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gamma_test import daily_from, intraday  # noqa: E402
from whale import rth_mask  # noqa: E402

LEVELS = "data/opra/ndx_gamma_levels.parquet"
NQ = "data/mbo/NQ_minutes.parquet"


def load():
    lv = pd.read_parquet(LEVELS)
    lv.index = pd.DatetimeIndex(lv.index)
    nq = pd.read_parquet(NQ)
    nq = nq[rth_mask(nq.index)].copy()
    nq["date"] = pd.DatetimeIndex(nq.index).normalize()

    ndx_daily = daily_from(intraday())
    nq_daily = nq.groupby("date")["close"].last()
    # Empirical basis: futures over index, on each shared date.
    basis = (nq_daily / ndx_daily["close"]).dropna()
    lv = lv.join(basis.rename("basis"), how="inner").dropna(subset=["basis"])
    for c in ("call_wall", "put_wall", "gamma_flip", "spot"):
        lv[c + "_nq"] = lv[c] * lv["basis"]
    return lv, nq


def touches(lv, nq, hold_min=30):
    """First time each session reaches a wall, and what happens next."""
    out = []
    for d, g in nq.groupby("date"):
        if d not in lv.index:
            continue
        row = lv.loc[d]
        g = g.sort_index()
        op = float(g["close"].iloc[0])
        for side, lvl, name in ((-1, row["call_wall_nq"], "call"),
                                (+1, row["put_wall_nq"], "put")):
            if not np.isfinite(lvl):
                continue
            # Only a level the session has to travel towards is a test of it.
            if side < 0 and lvl <= op:
                continue
            if side > 0 and lvl >= op:
                continue
            hit = (g["high"] >= lvl) if side < 0 else (g["low"] <= lvl)
            if not hit.any():
                continue
            i = int(np.argmax(hit.to_numpy()))
            t0 = g.index[i]
            fwd = g[g.index <= t0 + pd.Timedelta(minutes=hold_min)]
            if len(fwd) < 2:
                continue
            # Fade the level: short the call wall, long the put wall.
            pnl = side * (float(fwd["close"].iloc[-1]) - lvl)
            out.append({"date": d, "side": name, "level": lvl, "t0": t0,
                        "pnl_pts": pnl, "pnl_pct": pnl / lvl,
                        "to_close": side * (float(g["close"].iloc[-1]) - lvl) / lvl})
    return pd.DataFrame(out)


def control(lv, nq, pct=0.005, hold_min=30):
    """The same trade at a level with no gamma content: a fixed % from the open."""
    out = []
    for d, g in nq.groupby("date"):
        if d not in lv.index:
            continue
        g = g.sort_index()
        op = float(g["close"].iloc[0])
        for side, lvl in ((-1, op * (1 + pct)), (+1, op * (1 - pct))):
            hit = (g["high"] >= lvl) if side < 0 else (g["low"] <= lvl)
            if not hit.any():
                continue
            i = int(np.argmax(hit.to_numpy()))
            t0 = g.index[i]
            fwd = g[g.index <= t0 + pd.Timedelta(minutes=hold_min)]
            if len(fwd) < 2:
                continue
            pnl = side * (float(fwd["close"].iloc[-1]) - lvl)
            out.append({"pnl_pct": pnl / lvl})
    return pd.DataFrame(out)


def stat(x, label, point=20.0, ref=25000.0):
    x = pd.Series(x).dropna()
    if len(x) < 15:
        return {"label": label, "n": len(x)}
    m, s = float(x.mean()), float(x.std(ddof=1))
    return {"label": label, "n": len(x), "bp": m * 1e4,
            "usd": m * ref * point,
            "t": m / (s / np.sqrt(len(x))) if s > 0 else np.nan,
            "hit": float((x > 0).mean())}


def fmt(rows):
    hdr = f"{'trade':<34s}{'n':>6s}{'bp':>9s}{'$/trade':>10s}{'hit%':>7s}{'t':>7s}"
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        if r.get("n", 0) < 15:
            out.append(f"{r['label']:<34s}{r.get('n',0):>6d}   (too few)")
            continue
        out.append(f"{r['label']:<34s}{r['n']:>6d}{r['bp']:>9.2f}{r['usd']:>10.2f}"
                   f"{r['hit']*100:>7.1f}{r['t']:>7.2f}")
    return "\n".join(out)


def main():
    lv, nq = load()
    print(f"{len(lv)} sessions with both option levels and NQ minute bars, "
          f"{lv.index.min().date()} .. {lv.index.max().date()}")
    print(f"median NQ/NDX basis {lv['basis'].median():.4f}\n")

    for hold in (15, 30, 60):
        t = touches(lv, nq, hold_min=hold)
        if t.empty:
            print(f"hold {hold} min: no touches")
            continue
        c = control(lv, nq, pct=0.005, hold_min=hold)
        print(f"=== fade the level, hold {hold} minutes ===")
        print(fmt([
            stat(t[t["side"] == "call"]["pnl_pct"], "  short the call wall"),
            stat(t[t["side"] == "put"]["pnl_pct"], "  long the put wall"),
            stat(t["pnl_pct"], "  both walls"),
            stat(c["pnl_pct"], "  control: fade 0.5% from open"),
        ]))
        print()

    t = touches(lv, nq, hold_min=30)
    half = lv.index[len(lv) // 2]
    print("=== both walls, 30-minute hold, split by date ===")
    print(fmt([
        stat(t[pd.DatetimeIndex(t["date"]) <= half]["pnl_pct"], "  first half"),
        stat(t[pd.DatetimeIndex(t["date"]) > half]["pnl_pct"], "  second half"),
    ]))


if __name__ == "__main__":
    main()
