"""Settle the stage-risk question on the development window with many starts.

The single-path comparison split three ways depending on which strategy you
looked at, which is what one sample of a noisy process looks like. This runs
every scheme across many challenge start dates so the decision rests on a
distribution rather than on one lucky or unlucky path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import month_starts, pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402

import strategies_final as F  # noqa: E402

SCHEMES = {
    "flat": {"phase1": 1.0, "phase2": 1.0, "funded": 1.0},
    "push": {"phase1": 1.35, "phase2": 1.35, "funded": 1.0},
    "push/ease": {"phase1": 1.35, "phase2": 1.35, "funded": 0.70},
}

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    starts = month_starts(df, "2025-02-01", "2025-05-05", every_days=5)
    print(f"Stage-risk decision on DEV: {len(starts)} starts x 205 days")
    print("(all runs end before the Dec 2025 test window)\n")
    print(f"{'strategy':24s} {'scheme':>10s} {'P1%':>5s} {'fund%':>6s} {'alive%':>7s} "
          f"{'p1d':>5s} {'payout$':>8s}")
    totals = {k: 0.0 for k in SCHEMES}
    for cls in F.FINAL:
        for sname, scheme in SCHEMES.items():
            res = pass_rate_parallel("strategies_final", cls.__name__,
                                     {"stage_risk": scheme}, starts, 205)
            if not res:
                continue
            totals[sname] += res["avg_payout"]
            print(f"{cls.name[:24]:24s} {sname:>10s} {res['p1_pass'] * 100:5.0f} "
                  f"{res['funded'] * 100:6.0f} {res['funded_alive'] * 100:7.0f} "
                  f"{res['med_p1_days']:5.0f} {res['payout_when_funded']:8.0f}", flush=True)
    print("\nmean payout per run, summed over the four accounts:")
    for k, v in totals.items():
        print(f"  {k:>10s}  ${v:,.0f}")
