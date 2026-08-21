"""How concentrated is A3's profit, really?

The consistency rule caps the best single day at 40% of the profit being
withdrawn. This measures the actual share, so the gap between "just misses" and
"nowhere close" is visible -- that difference decides whether the strategy can
be adjusted or has to be replaced for this account type.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market, Rules, run_challenge  # noqa: E402
from evaluate import month_starts  # noqa: E402
from run import load, slice_period  # noqa: E402

import strategies_final as F  # noqa: E402


def day_pnl_series(stage, rules):
    rows = {}
    for t in stage.trades:
        d = pd.Timestamp(t.ts_out).normalize()
        rows[d] = rows.get(d, 0.0) + t.pnl
    return rows


if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    mkt = Market(df)
    rules = Rules()
    shares = []
    print("funded stages: profit concentration at the 21-day mark\n")
    print(f"{'start':12s} {'days':>5s} {'profit$':>9s} {'best day$':>10s} "
          f"{'share':>7s} {'passes 40%?':>12s}")
    for a, b in [("2025-02-01", "2025-05-05"), ("2025-12-01", "2026-02-20")]:
        for s in month_starts(df, a, b, every_days=5):
            e = int(np.searchsorted(mkt.ts, mkt.ts[s] + np.timedelta64(190, "D")))
            e = min(e, mkt.n)
            if (mkt.ts[e - 1] - mkt.ts[s]) / np.timedelta64(1, "D") < 20:
                continue
            stages = run_challenge(mkt, F.A3_DonchianH4(), rules, s, e)
            if len(stages) < 3:
                continue
            fu = stages[2]
            dp = day_pnl_series(fu, rules)
            if not dp:
                continue
            total = sum(dp.values())
            best = max(dp.values())
            if total <= 0:
                continue
            share = best / total
            shares.append(share)
            print(f"{str(pd.Timestamp(mkt.ts[s]).date()):12s} "
                  f"{fu.calendar_days:5.0f} {total:9.0f} {best:10.0f} "
                  f"{share * 100:6.0f}% {'YES' if share <= 0.40 else 'no':>12s}")
    if shares:
        arr = np.array(shares)
        print(f"\nfunded runs with positive profit: {len(arr)}")
        print(f"  best-day share  median {np.median(arr) * 100:.0f}%  "
              f"min {arr.min() * 100:.0f}%  max {arr.max() * 100:.0f}%")
        print(f"  would clear the 40% rule: {(arr <= 0.40).mean() * 100:.0f}%")
