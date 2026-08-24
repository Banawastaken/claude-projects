"""Sweep the PEAD parameters on the design window, then test the winner once."""

from __future__ import annotations

import itertools
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from multistrat import stats  # noqa: E402
from pead_concordant import build, sides  # noqa: E402
from pead_sweep import portfolio, price_paths, walk  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")

GRID = {
    "hold":       [20, 40, 60, 90],
    "side":       ["both", "long", "short"],
    "min_surp":   [0.0, 0.02, 0.05, 0.10],
    "weight":     [0.05, 0.10, 0.20],
    "hard_stop":  [None, 0.05, 0.10, 0.15],
    "trail_stop": [None, 0.05, 0.10],
}


def prepare():
    df, rets, sessions = build(hold=90, market_adjust=False)
    C, H, L = price_paths(sorted(df["ticker"].unique()))
    C = C.reindex(sessions).ffill()
    H = H.reindex(sessions).ffill()
    L = L.reindex(sessions).ffill()
    import os
    m = pd.read_parquet("data/yahoo/SPY.parquet")
    mkt = pd.Series(m["adj_close"].values,
                    index=pd.DatetimeIndex(m["date"]).tz_localize(None)).pct_change()
    return df, C, H, L, sessions, mkt


def run_one(df, C, H, L, sessions, mkt, hold, side, min_surp, weight,
            hard_stop, trail_stop, window=None):
    d = df.copy()
    d["side"] = sides(d, concordant=True)
    if min_surp > 0:
        sp = d["surprise_pct"].abs()
        d.loc[~(sp >= min_surp), "side"] = 0.0
    if side == "long":
        d.loc[d["side"] < 0, "side"] = 0.0
    elif side == "short":
        d.loc[d["side"] > 0, "side"] = 0.0
    tr_in = d[d["side"] != 0]
    if window is not None:
        lo, hi = window
        dates = pd.DatetimeIndex(sessions)[tr_in["entry_i"].to_numpy()]
        tr_in = tr_in[(dates >= lo) & (dates < hi)]
    if len(tr_in) < 40:
        return None, 0
    tr = walk(tr_in, C, H, L, hold, hard_stop, trail_stop)
    if tr.empty:
        return None, 0
    r = portfolio(tr, sessions, weight, mkt=mkt)
    if window is not None:
        r = r[(r.index >= window[0]) & (r.index < window[1])]
    return r, len(tr)


def main():
    df, C, H, L, sessions, mkt = prepare()
    print(f"{len(df):,} announcements, {df['ticker'].nunique()} names, "
          f"{df['date'].min()} .. {df['date'].max()}")
    keys = list(GRID)
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"{len(combos):,} parameter combinations, scored on the design window "
          f"only (2015-2020)\n")

    des = (pd.Timestamp.min, SPLIT)
    rows = []
    for c in combos:
        kw = dict(zip(keys, c))
        r, n = run_one(df, C, H, L, sessions, mkt, window=des, **kw)
        if r is None:
            continue
        s = stats(r)
        if not s or s.get("n", 0) < 100 or not np.isfinite(s.get("sharpe", np.nan)):
            continue
        rows.append({**kw, "trades": n, "sharpe": s["sharpe"], "cagr": s["cagr"],
                     "maxdd": s["maxdd"], "t": s["t"]})
    res = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(f"{len(res):,} combinations produced a usable design-window result\n")

    print("=" * 78)
    print("TOP 10 ON THE DESIGN WINDOW  (this is the selection, not the answer)")
    print("=" * 78)
    show = ["hold", "side", "min_surp", "weight", "hard_stop", "trail_stop",
            "trades", "sharpe", "cagr", "maxdd"]
    print(res.head(10)[show].to_string(index=False))

    print("\n" + "=" * 78)
    print("WHAT EACH PARAMETER IS WORTH, AVERAGED OVER EVERYTHING ELSE")
    print("=" * 78)
    for k in keys:
        g = res.groupby(res[k].astype(str))["sharpe"].agg(["mean", "median", "count"])
        print(f"\n  {k}:")
        for v, row in g.iterrows():
            print(f"    {v:<8s} mean {row['mean']:+.3f}  median {row['median']:+.3f}"
                  f"  ({int(row['count'])} cells)")

    print("\n" + "=" * 78)
    print("THE DESIGN WINNER, RUN ONCE ON THE HOLDOUT")
    print("=" * 78)
    best = res.iloc[0]
    kw = {k: (None if pd.isna(best[k]) else best[k]) for k in keys}
    for k in ("hold",):
        kw[k] = int(kw[k])
    print(f"  chosen: {kw}")
    for tag, w in (("design 2015-2020", des),
                   ("holdout 2021-2026", (SPLIT, pd.Timestamp.max)),
                   ("full sample", None)):
        r, n = run_one(df, C, H, L, sessions, mkt, window=w, **kw)
        if r is None:
            print(f"  {tag}: no trades")
            continue
        s = stats(r)
        print(f"  {tag:<20s} {n:>4d} trades  Sharpe {s['sharpe']:5.2f}  "
              f"CAGR {s['cagr']*100:6.2f}%  maxDD {s['maxdd']*100:6.2f}%  "
              f"t {s['t']:5.2f}")

    print("\n" + "=" * 78)
    print("HOLDOUT PERFORMANCE OF THE TOP 20 DESIGN CELLS")
    print("=" * 78)
    hold_sh = []
    for _, row in res.head(20).iterrows():
        kw = {k: (None if pd.isna(row[k]) else row[k]) for k in keys}
        kw["hold"] = int(kw["hold"])
        r, n = run_one(df, C, H, L, sessions, mkt,
                       window=(SPLIT, pd.Timestamp.max), **kw)
        s = stats(r) if r is not None else None
        hold_sh.append(s["sharpe"] if s else np.nan)
    hs = np.array(hold_sh, dtype=float)
    print(f"  design Sharpe of these 20: {res.head(20)['sharpe'].mean():.2f} mean")
    print(f"  holdout Sharpe of the same 20: {np.nanmean(hs):.2f} mean, "
          f"{np.nanmin(hs):.2f} to {np.nanmax(hs):.2f}")
    print(f"  {int(np.sum(hs > 0))} of {len(hs)} stayed positive out of sample")
    res.to_csv("data/pead/sweep.csv", index=False)
    print("\nwrote data/pead/sweep.csv")


if __name__ == "__main__":
    main()
