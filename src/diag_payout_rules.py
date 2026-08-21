"""Would A3's payouts actually clear FundedNext's payout gates?

Two rules were missing from the original model, and both bite exactly the kind
of strategy A3 is:

* a reward request needs at least 2% account growth;
* the 40% consistency rule caps the best single day at 40% of the profit being
  withdrawn.

A3 takes roughly one trade a day at most and runs wide targets, so a single
good day can easily be more than 40% of a modest profit. This compares payouts
with the gates off and on.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import month_starts, pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402

WINDOWS = [("DEV", "2025-02-01", "2025-05-05", 205),
           ("TEST", "2025-12-01", "2026-02-20", 175)]

GATES = {
    "gates off": {"payout_min_growth": 0.0, "payout_max_day_share": 9.99},
    "2% only": {"payout_min_growth": 0.02, "payout_max_day_share": 9.99},
    "2% + 40% consistency": {"payout_min_growth": 0.02, "payout_max_day_share": 0.40},
}

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    print("A3 on gold: effect of the real payout gates\n")
    print(f"{'window':6s} {'gates':22s} {'funded%':>8s} {'alive%':>7s} "
          f"{'paid$/run':>10s} {'paid$ if funded':>16s}")
    for label, a, b, horizon in WINDOWS:
        starts = month_starts(df, a, b, every_days=5)
        for gname, over in GATES.items():
            r = pass_rate_parallel("strategies_final", "A3_DonchianH4", {}, starts,
                                   horizon, rule_over=over)
            if not r:
                continue
            print(f"{label:6s} {gname:22s} {r['funded'] * 100:8.0f} "
                  f"{r['funded_alive'] * 100:7.0f} {r['avg_payout']:10.0f} "
                  f"{r['payout_when_funded']:16.0f}", flush=True)
    print("\nA challenge attempt costs $59.99, refunded with the first reward.")
