"""Follow-ups to the overnight study: what actually decides whether it is real.

Three questions, in the order that matters:
  1. how much of the result is the financing assumption rather than the market;
  2. does the Tue/Wed weekday pattern survive out of sample, or is it the
     product of having looked at four weekdays and kept the best two;
  3. is there a better exit hour, or does the whole surface look like noise.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from anomaly import financing_bp, fmt, load, overnight_legs, summarise  # noqa: E402

DESIGN_END = pd.Timestamp("2021-01-01", tz="America/New_York")


def main():
    for name in ("NDX100", "SPX500"):
        df = load(name)
        on = overnight_legs(df)
        print(f"================ {name} ================")
        print(f"{len(on):,} overnight holds. Friday-evening entries are absent by "
              f"construction:\nthe 36h pairing window excludes weekend holds "
              f"(3 nights of financing, gap risk).\n")

        print("1. sensitivity to the financing rate (the assumption doing the work)")
        rows = []
        for rate in (0.0, 0.02, 0.04, 0.05, 0.065, 0.08):
            r = on["net_bp"] - financing_bp(rate)
            rows.append(summarise(r, f"  financing {rate*100:>4.1f}%/yr"))
        print(fmt(rows))
        be = on["net_bp"].mean() / 1e4 * 360 * 100
        print(f"  -> break-even financing rate: {be:.2f}%/yr "
              f"(above this the hold loses money after spread)\n")

        print("2. weekday pattern, design vs holdout, after spread + financing")
        on["net_fin"] = on["net_bp"] - financing_bp()
        d = pd.to_datetime(on["date"])
        des, hold = on[d < DESIGN_END], on[d >= DESIGN_END]
        names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
        rows = []
        for dow in sorted(on["dow"].unique()):
            a = des[des["dow"] == dow]["net_fin"]
            b = hold[hold["dow"] == dow]["net_fin"]
            if len(a) < 20 or len(b) < 20:
                continue
            rows.append(summarise(a, f"  exit {names.get(dow, dow)}  design"))
            rows.append(summarise(b, f"  exit {names.get(dow, dow)}  holdout"))
        print(fmt(rows))

        tw = on[on["dow"].isin([1, 2])]
        d2 = pd.to_datetime(tw["date"])
        print("\n  the Tue+Wed subset on its own:")
        print(fmt([summarise(tw[d2 < DESIGN_END]["net_fin"], "  Tue+Wed design"),
                   summarise(tw[d2 >= DESIGN_END]["net_fin"], "  Tue+Wed holdout")]))

        print("\n3. exit-hour scan (entry fixed at 16:00 NY), after spread + financing")
        rows = []
        for hr in (0, 2, 4, 6, 7, 8, 9, 10):
            o = overnight_legs(df, close_hour=15, open_hour=hr)
            if len(o) < 200:
                continue
            rows.append(summarise(o["net_bp"] - financing_bp(),
                                  f"  exit {hr:02d}:00 NY"))
        print(fmt(rows))
        print()


if __name__ == "__main__":
    main()
