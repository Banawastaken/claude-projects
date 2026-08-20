"""Does pushing harder in the phases and easing off once funded actually help?

Checked on the development window as well as the test window, because a change
that only helps in 2026 is fitted to 2026.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diag_throttle import journey, variant  # noqa: E402
from engine import Market, Rules  # noqa: E402
from run import load, slice_period  # noqa: E402

import strategies_final as F  # noqa: E402

SCHEMES = {
    "flat": {"phase1": 1.0, "phase2": 1.0, "funded": 1.0},
    "push/ease": {"phase1": 1.35, "phase2": 1.35, "funded": 0.70},
    "push only": {"phase1": 1.35, "phase2": 1.35, "funded": 1.0},
    "ease only": {"phase1": 1.0, "phase2": 1.0, "funded": 0.70},
}

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    mkt = Market(df)
    rules = Rules()
    for label, a, b in [("DEV", "2025-02-01", "2025-12-01"), ("TEST", "2025-12-01", None)]:
        i0, i1 = slice_period(df, a, b)
        print(f"\n=== {label} window")
        print(f"{'strategy':22s} {'scheme':>10s} {'P1':>3s} {'P2':>3s} {'FUND':>5s} "
              f"{'ALIVE':>6s} {'pays':>5s} {'paid$':>8s} {'final$':>9s} {'maxDD%':>7s}")
        for cls in F.FINAL:
            for sname, scheme in SCHEMES.items():
                c = variant(cls, stage_risk=scheme)
                r = journey(mkt, c, rules, i0, i1)
                print(f"{cls.name[:22]:22s} {sname:>10s} {str(r['p1'])[0]:>3s} "
                      f"{str(r['p2'])[0]:>3s} {str(r['funded'])[0]:>5s} "
                      f"{str(r['alive'])[0]:>6s} {r['payouts']:5d} "
                      f"{r['payout_usd']:8.0f} {r['final']:9.0f} {r['maxdd']:7.2f}",
                      flush=True)
