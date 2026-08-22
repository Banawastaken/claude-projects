"""The one gold calendar cell that survived the holdout: January.

The scan found January positive on both windows with the same sign and a
similar size.  Before treating that as an edge it has to clear three further
hurdles, which is what this does:

  1. is it bigger than gold's unconditional drift, or just gold going up;
  2. is it carried by the whole sample or by one or two Januaries;
  3. does it survive being traded -- sized against a prop account's drawdown
     limits, with the spread paid, held for a month with gold's actual
     within-month excursions.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from anomaly import NY, fmt, load, summarise  # noqa: E402
from study_gold_calendar import daily_returns  # noqa: E402


def main():
    df = load("XAUUSD")
    d = daily_returns(df)
    jan = d[d["month"] == 1]["gross_bp"]
    rest = d[d["month"] != 1]["gross_bp"]

    print("1. January against the rest of the year, daily close-to-close, gross")
    print(fmt([summarise(jan, "  January"),
               summarise(rest, "  all other months"),
               summarise(d["gross_bp"], "  unconditional")]))
    diff = jan.mean() - rest.mean()
    se = np.sqrt(jan.var(ddof=1) / len(jan) + rest.var(ddof=1) / len(rest))
    print(f"  difference {diff:+.2f} bp/day, t = {diff/se:.2f}  "
          f"(Welch, {len(jan)} vs {len(rest)} days)")
    print(f"  -> January is {jan.mean()/max(rest.mean(),1e-9):.1f}x the "
          f"other-month daily mean\n")

    print("2. January by year -- is it broad, or one or two outliers?")
    print(f"  {'year':<8s}{'days':>6s}{'return %':>11s}{'max DD %':>11s}")
    print("  " + "-" * 34)
    rows = []
    for yr, g in d[d["month"] == 1].groupby(d[d["month"] == 1].index.year):
        r = g["gross_bp"] / 1e4
        cum = (1 + r).cumprod()
        dd = (cum / cum.cummax() - 1).min() * 100
        tot = (cum.iloc[-1] - 1) * 100
        rows.append((yr, len(g), tot, dd))
        print(f"  {yr:<8d}{len(g):>6d}{tot:>11.2f}{dd:>11.2f}")
    pos = sum(1 for r in rows if r[2] > 0)
    arr = np.array([r[2] for r in rows])
    print("  " + "-" * 34)
    print(f"  {pos}/{len(rows)} Januaries positive   "
          f"mean {arr.mean():+.2f}%   median {np.median(arr):+.2f}%   "
          f"worst {arr.min():+.2f}%")
    trimmed = np.sort(arr)[1:-1]
    print(f"  dropping the best and worst year: mean {trimmed.mean():+.2f}% "
          f"({len(trimmed)} years)\n")

    print("3. traded as a prop position: long gold for January, sized so the")
    print("   worst historical January drawdown would not breach the account")
    spread_bp = ((df['ask_open'] - df['open']) / df['open'] * 1e4).median()
    worst_dd = min(r[3] for r in rows)
    print(f"   worst January drawdown in the sample: {worst_dd:.2f}%")
    for limit, tag in ((5.0, "daily loss cap"), (10.0, "static max loss")):
        lev = limit / abs(worst_dd)
        print(f"   to keep the worst January inside the {limit:.0f}% {tag}: "
              f"leverage {lev:.2f}x")
        print(f"     -> mean January P&L at that size: "
              f"{arr.mean()*lev:+.2f}%  (median {np.median(arr)*lev:+.2f}%)")
    print(f"\n   round-trip spread on the single entry/exit: {spread_bp*2:.1f} bp "
          f"({spread_bp*2/100:.3f}%) -- negligible over a month-long hold,")
    print("   unlike the hourly scan where it was the whole result.")

    print("\n4. what this is worth against an 8% phase-1 target")
    need = 8.0
    lev = 10.0 / abs(worst_dd)
    print(f"   at {lev:.2f}x, January alone delivers a mean {arr.mean()*lev:+.2f}% "
          f"and hits +{need:.0f}% in "
          f"{sum(1 for a in arr if a*lev >= need)}/{len(arr)} years.")
    print("   It is one trade per year: it cannot be the strategy, only a part of one.")


if __name__ == "__main__":
    main()
