"""Does waiting longer between reward requests beat the consistency rule?

At 21 days A3's profit sits in one or two trades, so the best day is most of it
and the 40% rule blocks the withdrawal. Rolling-window analysis says that by 90
days the median best-day share falls to 35%, under the cap. This tests whether
that translates into actual payouts once the account also has to stay alive and
in profit the whole time.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import month_starts, pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402

CADENCES = [(21, 14), (45, 30), (60, 45), (90, 60), (120, 90)]
WINDOWS = [("DEV", "2025-02-01", "2025-05-05", 205),
           ("TEST", "2025-12-01", "2026-02-20", 175)]

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    print("A3 on gold, full payout gates on (2% growth + 40% consistency)\n")
    print(f"{'window':6s} {'cadence':>10s} {'funded%':>8s} {'alive%':>7s} "
          f"{'payouts/run':>12s} {'paid$ if funded':>16s}")
    for label, a, b, horizon in WINDOWS:
        starts = month_starts(df, a, b, every_days=5)
        for first, nxt in CADENCES:
            r = pass_rate_parallel(
                "strategies_final", "A3_DonchianH4", {}, starts, horizon,
                rule_over={"payout_first_days": first, "payout_next_days": nxt},
            )
            if not r:
                continue
            npay = r["detail"]["n_payouts"].mean()
            print(f"{label:6s} {f'{first}/{nxt}':>10s} {r['funded'] * 100:8.0f} "
                  f"{r['funded_alive'] * 100:7.0f} {npay:12.2f} "
                  f"{r['payout_when_funded']:16.0f}", flush=True)
