"""How much does running on H1 bars instead of M1 change the answer?

The multi-instrument study uses hourly bars, because minute data for thirty
symbols is thousands of downloads. Gold has both, so the size of that
approximation can be measured rather than assumed.

Coarser bars cut both ways: entries fill at the next hourly open instead of the
next minute (worse, and more realistic for a strategy this slow), while
intrabar equity is sampled at hourly extremes rather than minute extremes,
which can understate a drawdown that spiked and recovered inside one hour.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market  # noqa: E402
from evaluate import pass_rate_parallel  # noqa: E402
from run import load  # noqa: E402


def m1_to_h1(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the minute series to hourly, matching the H1 feed's shape."""
    d = df.copy()
    d["bucket"] = d["minute"] // 60
    g = d.groupby("bucket", sort=True)
    out = g.agg(
        ts=("ts", "first"),
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        ask_open=("ask_open", "first"), ask_high=("ask_high", "max"),
        ask_low=("ask_low", "min"), ask_close=("ask_close", "last"),
        volume=("volume", "sum"),
    ).reset_index(drop=True)
    out["minute"] = out["ts"].astype("int64") // 10**9 // 60
    out["spread_med"] = out["ask_close"] - out["close"]
    return out


WINDOWS = [("DEV", "2025-02-01", "2025-05-05", 205),
           ("TEST", "2025-12-01", "2026-02-20", 175)]

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    h1 = m1_to_h1(df)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "instruments")
    os.makedirs(out, exist_ok=True)
    h1.to_parquet(os.path.join(out, "_XAUUSD_H1_FROM_M1.parquet"), index=False)
    print(f"M1 bars {len(df):,} -> H1 bars {len(h1):,}\n")

    from evaluate import month_starts

    print(f"{'window':6s} {'bars':>5s} {'P1%':>5s} {'fund%':>6s} {'alive%':>7s} "
          f"{'breach%':>8s} {'ddMed':>6s} {'wdMax':>6s} {'trades':>7s} {'payout$':>8s}")
    for label, a, b, horizon in WINDOWS:
        for tag, frame in (("M1", df), ("H1", h1)):
            starts = month_starts(frame, a, b, every_days=5)
            t = "xauusd_m1_clean" if tag == "M1" else "INSTRUMENT:_XAUUSD_H1_FROM_M1"
            r = pass_rate_parallel("strategies_final", "A3_DonchianH4", {}, starts,
                                   horizon, tag=t)
            if not r:
                continue
            print(f"{label:6s} {tag:>5s} {r['p1_pass'] * 100:5.0f} {r['funded'] * 100:6.0f} "
                  f"{r['funded_alive'] * 100:7.0f} {r['breach_rate'] * 100:8.0f} "
                  f"{r['dd_med']:6.2f} {r['worstday_max']:6.2f} "
                  f"{r['trades_med']:7.0f} {r['payout_when_funded']:8.0f}", flush=True)
