"""Audition candidates for the fourth account after Keltner failed the rules test.

Keltner had a respectable raw edge (PF 1.48) but only passed Phase 1 in 64% of
development starts and got funded in at most 18%. Raw edge and rule survival
are different questions; this picks the replacement on the second one.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import month_starts, pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402

# (label, module, class, param overrides)
CANDIDATES = [
    ("VolBreakout k.6 atr3", "strategies_v3", "VolBreakout",
     {"k_open": 0.6, "atr_mult": 3.0, "tp_r": 5.0}),
    ("VolBreakout k.45 atr2.5", "strategies_v3", "VolBreakout",
     {"k_open": 0.45, "atr_mult": 2.5, "tp_r": 5.0}),
    ("DonchH1 lb90 atr3", "strategies_v2", "DonchianV2",
     {"lookback": 90, "atr_mult": 3.0, "tp_r": 5.0, "adx_min": 18, "trend_filter": False,
      "max_trades_day": 2, "max_losses_day": 2}),
    ("DonchH4 lb20 atr1 tp4", "strategies_v4", "DonchianH4",
     {"lookback": 20, "atr_mult": 1.0, "tp_r": 4.0, "adx_min": 15}),
    ("Pullback atr3 tp4", "strategies_v2", "PullbackV2",
     {"atr_mult": 3.0, "tp_r": 4.0, "adx_min": 20, "max_trades_day": 2, "max_losses_day": 2}),
    ("Keltner b2.5 atr3 (incumbent)", "strategies_v3", "KeltnerBreak",
     {"band_k": 2.5, "atr_mult": 3.0, "tp_r": 5.0, "adx_min": 0,
      "max_trades_day": 2, "max_losses_day": 2}),
]

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    starts = month_starts(df, "2025-02-01", "2025-08-15", every_days=7)
    print(f"Slot-4 audition on DEV: {len(starts)} starts x 150 days\n")
    print(f"{'candidate':30s} {'risk':>5s} {'P1%':>5s} {'fund%':>6s} {'alive%':>7s} "
          f"{'p1d':>5s} {'payout$':>8s}")
    for label, mod, cls, params in CANDIDATES:
        for r in (0.0075, 0.010):
            p = dict(params)
            p["risk_pct"] = r
            res = pass_rate_parallel(mod, cls, p, starts, 150)
            if not res:
                continue
            print(f"{label[:30]:30s} {r * 100:4.2f}% {res['p1_pass'] * 100:5.0f} "
                  f"{res['funded'] * 100:6.0f} {res['funded_alive'] * 100:7.0f} "
                  f"{res['med_p1_days']:5.0f} {res['payout_when_funded']:8.0f}", flush=True)
