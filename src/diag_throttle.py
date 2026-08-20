"""Does the equity-curve risk throttle help, or is it fitted to 2026?

A change that only improves the test window is a fitted change. This measures
the throttle on the development window too; it has to at least not hurt there
before it earns a place.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market, Rules, run_challenge  # noqa: E402
from run import load, slice_period  # noqa: E402

import strategies_final as F  # noqa: E402

OFF = ((-99.0, 1.0),)  # throttle disabled


def variant(base, **params):
    return type(base.__name__ + "_v", (base,), {**params, "name": base.name})


def journey(mkt, cls, rules, i0, i1):
    stages = run_challenge(mkt, cls(), rules, i0, i1)
    got = len(stages) > 2
    funded = stages[2] if got else None
    return {
        "p1": stages[0].passed,
        "p2": len(stages) > 1 and stages[1].passed,
        "funded": got,
        "alive": bool(funded and not funded.breached),
        "payouts": len(funded.payouts) if funded else 0,
        "payout_usd": sum(p["net"] for p in funded.payouts) if funded else 0.0,
        "final": (funded or stages[-1]).final_balance,
        "maxdd": (funded or stages[-1]).max_dd_pct * 100,
    }


if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    mkt = Market(df)
    rules = Rules()
    windows = [("DEV", "2025-02-01", "2025-12-01"), ("TEST", "2025-12-01", None)]
    for label, a, b in windows:
        i0, i1 = slice_period(df, a, b)
        print(f"\n=== {label} window")
        print(f"{'strategy':24s} {'throttle':>9s} {'P1':>3s} {'P2':>3s} {'FUND':>5s} "
              f"{'ALIVE':>6s} {'pays':>5s} {'paid$':>8s} {'final$':>9s} {'maxDD%':>7s}")
        for cls in F.FINAL:
            for tag, lv in (("off", OFF), ("on", cls.throttle_levels)):
                c = variant(cls, throttle_levels=lv)
                r = journey(mkt, c, rules, i0, i1)
                print(f"{cls.name[:24]:24s} {tag:>9s} {str(r['p1'])[0]:>3s} "
                      f"{str(r['p2'])[0]:>3s} {str(r['funded'])[0]:>5s} "
                      f"{str(r['alive'])[0]:>6s} {r['payouts']:5d} "
                      f"{r['payout_usd']:8.0f} {r['final']:9.0f} {r['maxdd']:7.2f}",
                      flush=True)
