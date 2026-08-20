"""Strip non-tradeable minutes from the raw M1 feed.

Dukascopy publishes a bar for every minute of the calendar, including minutes
when the gold market is shut. Those filler bars carry zero volume and zero
range, and they are poison for indicators: 21% of the raw series is filler, and
Wilder's ATR decays toward zero across a weekend, so a Monday-morning ATR reads
a fraction of true volatility and every ATR-scaled stop comes out far too tight.

Dropping ticks-free minutes leaves only minutes the market actually traded.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    rng = df["high"] - df["low"]
    dead = (df["volume"] == 0) & (rng == 0)
    df = df[~dead].copy()

    # Guard against crossed or absurd quotes surviving from the feed.
    spread = df["ask_close"] - df["close"]
    bad = (spread < 0) | (spread > 15)
    df = df[~bad].copy()

    df = df.sort_values("ts").drop_duplicates("minute", keep="last").reset_index(drop=True)
    print(f"  {n0:,} -> {len(df):,} bars ({(n0 - len(df)) / n0 * 100:.1f}% removed)")
    return df


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "xauusd_m1"
    dst = sys.argv[2] if len(sys.argv) > 2 else "xauusd_m1_clean"
    df = pd.read_parquet(os.path.join(DATA, f"{src}.parquet"))
    print(f"Cleaning {src}")
    out = clean(df)
    out.to_parquet(os.path.join(DATA, f"{dst}.parquet"), index=False)
    rng = out["high"] - out["low"]
    print(f"Wrote {len(out):,} bars -> {dst}.parquet")
    print(f"  median bar range {rng.median():.3f}, median spread {(out['ask_close'] - out['close']).median():.3f}")
    print(f"  {out['ts'].min()} -> {out['ts'].max()}")
