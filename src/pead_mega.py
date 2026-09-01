"""The same drift rules on mega caps, at the threshold the scan picked.

Two things are being tested at once and they pull in opposite directions.
The 7.19% surprise threshold was the best of a thousand levels, and the null
said a shuffled parameter reaches it 18% of the time -- so it is expected to
travel badly. Mega caps are where the literature says the drift is smallest,
because post-earnings drift survives in proportion to how expensive a name is
to arbitrage, and nothing is cheaper to arbitrage than the twenty largest
companies in the market.

If the prediction is right this comes out near zero. Running it anyway is the
point: a forecast that is never checked is not a forecast, and if mega caps
come out strong then the arbitrage story is wrong and that is worth far more
than another confirmation.

The old basket is re-run alongside on identical code, so the comparison is
between universes and not between two scripts.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multistrat import fmt_stats, stats  # noqa: E402
from pead_concordant import backtest, build, sides  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")
BEST = 0.0719          # the level the two-decimal scan chose
DATA = "data/pead"


def run(events, label, pct_levels=(None, 0.05, BEST)):
    df, rets, sessions = build(events_path=events, hold=60, market_adjust=True)
    if df is None or df.empty:
        print(f"  {label}: no events")
        return {}

    p = pd.to_numeric(df["surprise_pct"], errors="coerce")
    print(f"\n  {label}")
    print(f"    names {df['ticker'].nunique():>3d}   announcements {len(df):>4d}"
          f"   median |surprise| {p.abs().median()*100:>5.1f}%"
          f"   at or above 7.19% {float((p.abs() >= BEST).mean())*100:>3.0f}%")

    out = {}
    for lv in pct_levels:
        side = sides(df, concordant=True, min_surprise_pct=lv)
        bt = backtest(df, rets, sessions, side, hold=60)
        if bt is None:
            continue
        name = "no filter" if lv is None else f">= {lv*100:.2f}%"
        out[name] = bt["ret"]
        out[name].attrs["n"] = int((side != 0).sum())
    return out


def show(title, series):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    rows = [(f"{k}  ({v.attrs['n']} trades)", stats(v)) for k, v in series.items()]
    print(fmt_stats(rows))


def main():
    mega = os.path.join(DATA, "mega_earnings.json")
    if not os.path.exists(mega):
        raise SystemExit("run the mega-cap fetch first")

    old = run(os.path.join(DATA, "av_earnings.json"), "the existing basket (large caps)")
    new = run(mega, "mega caps")

    show("WHOLE SAMPLE, market-adjusted, concordant rule, 60-day hold", 
         {f"large  {k}": v for k, v in old.items()} |
         {f"mega   {k}": v for k, v in new.items()})

    print("\n" + "=" * 96)
    print("DESIGN vs HOLDOUT")
    print("=" * 96)
    rows = []
    for tag, d in (("large", old), ("mega", new)):
        for k, v in d.items():
            rows.append((f"{tag} {k} design", stats(v[v.index < SPLIT])))
            rows.append((f"{tag} {k} holdout", stats(v[v.index >= SPLIT])))
    print(fmt_stats(rows))

    print("\n" + "=" * 96)
    print("THE PREDICTION, SCORED")
    print("=" * 96)
    for tag, d in (("large caps", old), ("mega caps", new)):
        for k, v in d.items():
            s = stats(v)
            print(f"  {tag:<11s} {k:<14s} Sharpe {s['sharpe']:>5.2f}   t {s['t']:>5.2f}"
                  f"   CAGR {s['cagr']*100:>5.2f}%")


if __name__ == "__main__":
    main()
