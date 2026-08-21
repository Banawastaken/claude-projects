"""Which of the four is actually the safest bet for a challenge?

"Most consistent and least likely to fail" is a question about the distribution
of outcomes, not about the single path already reported. This runs every
strategy from many challenge start dates across both the development and test
windows, and reports the risk profile: how often an account breached, how deep
drawdowns got, and how close the worst day came to the 5% daily limit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate import month_starts, pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402

import strategies_final as F  # noqa: E402

WINDOWS = [
    # label, first start, last start, horizon (days)
    ("DEV  2025", "2025-02-01", "2025-05-05", 205),
    ("TEST 2026", "2025-12-01", "2026-02-20", 175),
]

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    for label, a, b, horizon in WINDOWS:
        starts = month_starts(df, a, b, every_days=5)
        print(f"\n=== {label}: {len(starts)} challenge starts x {horizon} days")
        print(f"{'strategy':24s} {'P1%':>4s} {'fund%':>6s} {'alive%':>7s} {'breach%':>8s} "
              f"{'ddMed':>6s} {'ddP90':>6s} {'ddMax':>6s} {'wdMed':>6s} {'wdMax':>6s} "
              f"{'trades':>7s} {'payout$':>8s}")
        for cls in F.FINAL:
            r = pass_rate_parallel("strategies_final", cls.__name__, {}, starts, horizon)
            if not r:
                continue
            print(f"{cls.name[:24]:24s} {r['p1_pass'] * 100:4.0f} {r['funded'] * 100:6.0f} "
                  f"{r['funded_alive'] * 100:7.0f} {r['breach_rate'] * 100:8.0f} "
                  f"{r['dd_med']:6.2f} {r['dd_p90']:6.2f} {r['dd_max']:6.2f} "
                  f"{r['worstday_med']:6.2f} {r['worstday_max']:6.2f} "
                  f"{r['trades_med']:7.0f} {r['payout_when_funded']:8.0f}", flush=True)
    print("\nddMed/ddP90/ddMax = peak-to-trough drawdown %, median / 90th pct / worst")
    print("wdMed/wdMax       = worst single day %, against a 5% hard limit")
