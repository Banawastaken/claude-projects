"""Slot-4 decision on a longer horizon, without leaking test data.

AlignedMomentum's median Phase 1 took 127 days, so a 150-day horizon was
cutting it off mid-challenge rather than showing it failing. This re-runs the
contenders with a 205-day horizon.

The start dates are pulled back so that start + horizon still lands inside the
development period. Extending the horizon without moving the starts would have
run these simulations into Dec 2025 and beyond -- the test window -- and chosen
the fourth strategy using the data it is supposed to be judged on.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import month_starts, pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402

CANDIDATES = [
    ("Aligned atr2.5 tp4 adx25", "strategies_v5", "AlignedMomentum",
     {"atr_mult": 2.5, "tp_r": 4.0, "adx_min": 25}),
    ("DonchH4 lb20 atr1 tp4", "strategies_v4", "DonchianH4",
     {"lookback": 20, "atr_mult": 1.0, "tp_r": 4.0, "adx_min": 15}),
    ("A1 DonchianH1", "strategies_final", "A1_DonchianH1", {}),
    ("A2 Pullback", "strategies_final", "A2_TrendPullback", {}),
    ("A3 DonchianH4 lb10", "strategies_final", "A3_DonchianH4", {}),
]

HORIZON = 205

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    # last start must leave HORIZON days of runway before the test window opens
    starts = month_starts(df, "2025-02-01", "2025-05-05", every_days=5)
    print(f"Leak-free long-horizon check: {len(starts)} starts x {HORIZON} days")
    print("(all runs end before the Dec 2025 test window)\n")
    print(f"{'candidate':28s} {'risk':>5s} {'P1%':>5s} {'fund%':>6s} {'alive%':>7s} "
          f"{'p1d':>5s} {'payout$':>8s}")
    for label, mod, cls, params in CANDIDATES:
        for r in (0.0075, 0.010):
            p = dict(params)
            p["risk_pct"] = r
            res = pass_rate_parallel(mod, cls, p, starts, HORIZON)
            if not res:
                continue
            print(f"{label[:28]:28s} {r * 100:4.2f}% {res['p1_pass'] * 100:5.0f} "
                  f"{res['funded'] * 100:6.0f} {res['funded_alive'] * 100:7.0f} "
                  f"{res['med_p1_days']:5.0f} {res['payout_when_funded']:8.0f}", flush=True)
