"""Monte Carlo the four finalists across many challenge start dates (dev window)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import month_starts, pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402

import strategies_final as F  # noqa: E402

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    starts = month_starts(df, "2025-02-01", "2025-08-15", every_days=7)
    print(f"DEV Monte Carlo: {len(starts)} start dates x 150-day horizon\n")
    hdr = (f"{'strategy':26s} {'risk':>5s} {'P1%':>5s} {'P2%':>5s} {'fund%':>6s} "
           f"{'alive%':>7s} {'p1d':>5s} {'p2d':>5s} {'payout$':>8s}")
    print(hdr)
    for cls in F.FINAL:
        for r in (0.005, 0.0075, 0.010):
            res = pass_rate_parallel("strategies_final", cls.__name__, {"risk_pct": r},
                                     starts, 150)
            print(f"{cls.name[:26]:26s} {r * 100:4.2f}% {res['p1_pass'] * 100:5.0f} "
                  f"{res['p2_pass'] * 100:5.0f} {res['funded'] * 100:6.0f} "
                  f"{res['funded_alive'] * 100:7.0f} {res['med_p1_days']:5.0f} "
                  f"{res['med_p2_days']:5.0f} {res['payout_when_funded']:8.0f}", flush=True)
