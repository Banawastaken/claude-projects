"""Second slot-4 audition, and a re-check of the incumbents.

The intrabar breach-ordering fix changed the engine after the first Monte
Carlo, so the three incumbents are re-measured here rather than carried
forward on stale numbers.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import month_starts, pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402

CANDIDATES = [
    # incumbents, re-measured under the corrected engine
    ("A1 DonchianH1 (incumbent)", "strategies_final", "A1_DonchianH1", {}, (0.0075, 0.010)),
    ("A2 Pullback (incumbent)", "strategies_final", "A2_TrendPullback", {}, (0.0075, 0.010)),
    ("A3 DonchianH4 (incumbent)", "strategies_final", "A3_DonchianH4", {}, (0.0075, 0.010)),
    # slot-4 contenders
    ("Aligned atr2.5 tp4 adx25", "strategies_v5", "AlignedMomentum",
     {"atr_mult": 2.5, "tp_r": 4.0, "adx_min": 25}, (0.0075, 0.010)),
    ("Aligned atr2.5 tp6 adx25", "strategies_v5", "AlignedMomentum",
     {"atr_mult": 2.5, "tp_r": 6.0, "adx_min": 25}, (0.0075, 0.010)),
    ("Aligned atr1.5 tp3 adx25", "strategies_v5", "AlignedMomentum",
     {"atr_mult": 1.5, "tp_r": 3.0, "adx_min": 25}, (0.0075, 0.010)),
    ("DonchH4 lb20 atr1 tp4", "strategies_v4", "DonchianH4",
     {"lookback": 20, "atr_mult": 1.0, "tp_r": 4.0, "adx_min": 15}, (0.0075,)),
]

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    starts = month_starts(df, "2025-02-01", "2025-08-15", every_days=7)
    print(f"DEV Monte Carlo (corrected engine): {len(starts)} starts x 150 days\n")
    print(f"{'candidate':28s} {'risk':>5s} {'P1%':>5s} {'fund%':>6s} {'alive%':>7s} "
          f"{'p1d':>5s} {'p2d':>5s} {'payout$':>8s}")
    for label, mod, cls, params, risks in CANDIDATES:
        for r in risks:
            p = dict(params)
            p["risk_pct"] = r
            res = pass_rate_parallel(mod, cls, p, starts, 150)
            if not res:
                continue
            print(f"{label[:28]:28s} {r * 100:4.2f}% {res['p1_pass'] * 100:5.0f} "
                  f"{res['funded'] * 100:6.0f} {res['funded_alive'] * 100:7.0f} "
                  f"{res['med_p1_days']:5.0f} {res['med_p2_days']:5.0f} "
                  f"{res['payout_when_funded']:8.0f}", flush=True)
