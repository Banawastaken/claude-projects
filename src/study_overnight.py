"""Test the 'it pays while you sleep' claim: overnight vs intraday index drift.

The claim under test is the well-documented one that essentially all of the
equity index risk premium accrues outside the cash session.  It is tested here
the way it would have to be traded on a CFD prop account -- long the index from
the 16:00 NY close to the next 09:00 NY open, paying the spread both ways and
financing every night.

Windows follow RESEARCH_PROTOCOL.md: 2015-2020 to look at, 2021-2026 held back.
"""

from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "src")

from anomaly import (financing_bp, fmt, intraday_leg, load,  # noqa: E402
                     overnight_legs, summarise)

DESIGN_END = pd.Timestamp("2021-01-01", tz="America/New_York")
INSTRUMENTS = ["NDX100", "SPX500"]


def split(df, col="date"):
    d = pd.to_datetime(df[col])
    return df[d < DESIGN_END], df[d >= DESIGN_END]


def main():
    fin = financing_bp()
    print(f"assumed long CFD financing: {fin:.2f} bp per night "
          f"(6.5%/yr on notional)\n")

    for name in INSTRUMENTS:
        df = load(name)
        on = overnight_legs(df)
        intra = intraday_leg(df)
        on["after_fin_bp"] = on["net_bp"] - fin

        print(f"================ {name} ================")
        print(f"overnight pairs {len(on):,}   "
              f"median round-trip spread {on['spread_bp'].median():.2f} bp")

        rows = []
        for label, frame, col in (
                ("overnight  gross (mid-to-mid)", on, "gross_bp"),
                ("overnight  after spread", on, "net_bp"),
                ("overnight  after spread+financing", on, "after_fin_bp"),
                ("cash session  gross", intra, "gross_bp"),
                ("cash session  after spread", intra, "net_bp")):
            rows.append(summarise(frame[col], label))
        print(fmt(rows))

        des, hold = split(on)
        print("\n  design 2015-2020 vs holdout 2021-2026, after all costs:")
        print(fmt([summarise(des["after_fin_bp"], "  design  2015-2020"),
                   summarise(hold["after_fin_bp"], "  holdout 2021-2026")]))

        print("\n  by weekday (overnight, after all costs) -- "
              "entry evening -> exit next morning:")
        names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 6: "Sun"}
        rows = [summarise(g["after_fin_bp"], f"  exit {names.get(d, d)}")
                for d, g in on.groupby("dow") if len(g) > 20]
        print(fmt(rows))
        print()


if __name__ == "__main__":
    main()
