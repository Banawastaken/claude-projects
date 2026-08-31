"""What a minimum absolute surprise percentage actually does to the drift.

The published parameter is a percentage, and the earlier sweep tested a
currency amount by mistake: sides() compared the raw EPS difference, so
"0.05" there meant five cents rather than five percent. Five cents is a
trivial bar for a name earning nine dollars and an enormous one for a name
earning thirty, which makes that sweep's filter column close to meaningless
across a mixed basket. This measures the real thing.

The claim behind the filter is that a bigger surprise is a stronger signal, so
trading only the large ones should raise the return per trade even as it cuts
the count. Both halves are reported, because a filter that raises Sharpe purely
by discarding trades has not found anything -- it has just made the sample
smaller, and the error bar wider.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multistrat import fmt_stats, stats  # noqa: E402
from pead_concordant import backtest, build, sides  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")
LEVELS = [None, 0.0, 0.01, 0.02, 0.05, 0.10, 0.20]


def main():
    df, rets, sessions = build(hold=60, market_adjust=True)
    pct = pd.to_numeric(df["surprise_pct"], errors="coerce")

    print("=" * 74)
    print("THE SURPRISE DISTRIBUTION")
    print("=" * 74)
    print(f"  announcements with a usable estimate : {len(df)}")
    print(f"  of those, no computable percentage   : {int(pct.isna().sum())}"
          "   (estimate too near zero to divide by)")
    q = pct.abs().quantile([.25, .5, .75, .9, .95]).to_dict()
    print("  |surprise| as a percent of estimate:")
    for k, v in q.items():
        print(f"    {int(k*100):>3d}th percentile   {v*100:>8.1f}%")
    print(f"  share at or above 5%                 : "
          f"{float((pct.abs() >= 0.05).mean())*100:.0f}%")

    print()
    print("=" * 74)
    print("FILTERING ON IT, concordant rule, 60-day hold, market-adjusted")
    print("=" * 74)
    rows, keep = [], {}
    for lv in LEVELS:
        side = sides(df, concordant=True, min_surprise_pct=lv)
        bt = backtest(df, rets, sessions, side, hold=60)
        if bt is None:
            print(f"  >= {lv*100:.0f}%: no trades survive")
            continue
        n = int((side != 0).sum())
        label = "no filter" if lv is None else f">= {lv*100:>4.1f}%"
        rows.append((f"{label}  ({n} trades)", stats(bt["ret"])))
        keep[lv] = bt["ret"]
    print(fmt_stats(rows))

    print()
    print("=" * 74)
    print("THE 5% VERSION, split into the window it was designed on")
    print("and the window it was not")
    print("=" * 74)
    out = []
    for lv in (None, 0.05):
        r = keep[lv]
        label = "no filter" if lv is None else ">= 5%"
        out.append((f"{label}  design 15-20", stats(r[r.index < SPLIT])))
        out.append((f"{label}  holdout 21-26", stats(r[r.index >= SPLIT])))
    print(fmt_stats(out))

    print()
    print("=" * 74)
    print("IS THE FILTER PICKING BETTER TRADES, OR JUST FEWER?")
    print("=" * 74)
    side0 = sides(df, concordant=True)
    traded = df[side0 != 0].copy()
    traded["side"] = side0[side0 != 0]
    traded["pct"] = pd.to_numeric(traded["surprise_pct"], errors="coerce")

    # Forward return over the holding period, signed by the trade's direction,
    # so the question is whether a bigger surprise pays more per trade.
    px_i = traded["entry_i"].to_numpy()
    fwd = []
    for tk, i, sd in zip(traded["ticker"], px_i, traded["side"]):
        seg = rets[tk].iloc[i:i + 60]
        fwd.append(sd * float(seg.sum()) if len(seg) else np.nan)
    traded["fwd"] = fwd
    ok = traded.dropna(subset=["pct", "fwd"])

    print(f"  {'bucket':<22s}{'n':>6s}{'mean 60d':>11s}{'hit%':>8s}")
    print("  " + "-" * 47)
    edges = [(0, .01), (.01, .02), (.02, .05), (.05, .10), (.10, .25), (.25, 9e9)]
    for lo, hi in edges:
        m = (ok["pct"].abs() >= lo) & (ok["pct"].abs() < hi)
        if m.sum() < 5:
            print(f"  {lo*100:>4.0f}% to {hi*100:>5.0f}%      {int(m.sum()):>6d}   (too few)")
            continue
        print(f"  {lo*100:>4.0f}% to {hi*100:>5.0f}%      {int(m.sum()):>6d}"
              f"{ok.loc[m, 'fwd'].mean()*100:>10.2f}%{float((ok.loc[m,'fwd']>0).mean())*100:>7.0f}%")

    r = ok["pct"].abs().corr(ok["fwd"], method="spearman")
    n = len(ok)
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r ** 2)
    print(f"\n  rank correlation, |surprise%| vs signed 60-day return:"
          f" {r:+.3f}  (n={n}, t={t:+.2f})")
    print("  A filter on surprise size is only justified if this is positive.")


if __name__ == "__main__":
    main()
