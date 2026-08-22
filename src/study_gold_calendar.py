"""Calendar and session structure in spot gold.

The video title this responds to ("the 100-year-old gold strategy") does not
say which effect it means, so rather than guess a rule and then confirm it,
this scans the obvious calendar partitions and reports the whole surface --
including how many cells were looked at, which is what decides whether the
best one means anything.

Every partition is measured on 2015-2020 and then re-measured, unchanged, on
2021-2026.  A cell only counts if it survives that.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from anomaly import NY, fmt, load, summarise  # noqa: E402

DESIGN_END = pd.Timestamp("2021-01-01", tz=NY)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def hourly_returns(df):
    """Per-bar long return in bp, priced at the spread (in at ask, out at bid)."""
    out = df.copy()
    mid_o = 0.5 * (out["open"] + out["ask_open"])
    mid_c = 0.5 * (out["close"] + out["ask_close"])
    out["gross_bp"] = (mid_c / mid_o - 1.0) * 1e4
    out["net_bp"] = (out["close"] / out["ask_open"] - 1.0) * 1e4
    return out


def daily_returns(df):
    """One NY-day close-to-close observation, from H1 bars."""
    g = df.groupby("ny_date")
    d = pd.DataFrame({
        "close": g["close"].last(), "ask_close": g["ask_close"].last()})
    mid = 0.5 * (d["close"] + d["ask_close"])
    d["gross_bp"] = mid.pct_change() * 1e4
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    return d.dropna()


def two_window(x_des, x_hold, label):
    return [summarise(x_des, f"{label} design"), summarise(x_hold, f"{label} holdout")]


def main():
    df = load("XAUUSD")
    h = hourly_returns(df)
    d = daily_returns(df)
    des_h = h[h["ny_date"] < DESIGN_END]
    hold_h = h[h["ny_date"] >= DESIGN_END]
    des_d = d[d.index < DESIGN_END]
    hold_d = d[d.index >= DESIGN_END]

    print("XAUUSD 2015-2026, H1 bid/ask from the same feed used throughout.")
    print(f"{len(h):,} hourly bars, {len(d):,} daily observations.")
    print(f"median hourly spread {(h['ask_open']-h['open']).median():.3f} "
          f"({((h['ask_open']-h['open'])/h['open']*1e4).median():.2f} bp)\n")

    print("=" * 92)
    print("A. hour of day (NY), long the hour, net of spread -- 24 cells examined")
    print("=" * 92)
    rows = []
    for hr in range(24):
        a = des_h[des_h["ny_hour"] == hr]["net_bp"]
        b = hold_h[hold_h["ny_hour"] == hr]["net_bp"]
        if len(a) < 100 or len(b) < 100:
            continue
        ra, rb = summarise(a, f"  {hr:02d}:00 design"), summarise(b, f"  {hr:02d}:00 holdout")
        rows += [ra, rb]
    print(fmt(rows))

    print("\n" + "=" * 92)
    print("B. day of week, close-to-close, gross -- 5 cells examined")
    print("=" * 92)
    rows = []
    for i in range(5):
        a = des_d[des_d["dow"] == i]["gross_bp"]
        b = hold_d[hold_d["dow"] == i]["gross_bp"]
        if len(a) < 30 or len(b) < 30:
            continue
        rows += two_window(a, b, f"  {DOW[i]}")
    print(fmt(rows))

    print("\n" + "=" * 92)
    print("C. month of year, close-to-close, gross -- 12 cells examined")
    print("=" * 92)
    rows = []
    for m in range(1, 13):
        a = des_d[des_d["month"] == m]["gross_bp"]
        b = hold_d[hold_d["month"] == m]["gross_bp"]
        if len(a) < 30 or len(b) < 30:
            continue
        rows += two_window(a, b, f"  {MONTHS[m-1]}")
    print(fmt(rows))

    print("\n" + "=" * 92)
    print("D. how many cells would look good by chance")
    print("=" * 92)
    n_cells = 24 + 5 + 12
    print(f"  {n_cells} cells scanned. At the 5% level roughly "
          f"{n_cells*0.05:.1f} would clear t=2 on the design window alone")
    print("  with no effect present at all, and about "
          f"{n_cells*0.05*0.05:.2f} would clear it twice.")
    print("  A cell is only interesting here if it clears both windows in the")
    print("  same direction -- and even then the count above is the yardstick.")


if __name__ == "__main__":
    main()
