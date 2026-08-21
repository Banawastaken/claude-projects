"""Assemble decade parquets from whatever monthly files are already cached.

The upstream feed throttles hard, so this decouples assembly from fetching:
any instrument with enough cached months becomes usable immediately, and the
rest can be filled in later without redoing the work.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_h1 as F  # noqa: E402
from fetch_decade_all import START, END, build, divisor_for  # noqa: E402
from universe import UNIVERSE  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "decade")
MIN_MONTHS = 250  # of 280; a couple of missing months is not a problem


def cached_count(sym):
    d = os.path.join(F.CACHE, sym)
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".bi5")])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for inst in UNIVERSE:
        path = os.path.join(OUT, f"{inst.fn_name}.parquet")
        if os.path.exists(path):
            continue
        n = cached_count(inst.duka)
        if n < MIN_MONTHS:
            continue
        div = divisor_for(inst)
        if div is None:
            print(f"  {inst.fn_name:8s} skip (no divisor)", flush=True)
            continue
        df = build(inst, div)
        if df is None or len(df) < 5000:
            print(f"  {inst.fn_name:8s} skip (insufficient)", flush=True)
            continue
        df.to_parquet(path, index=False)
        print(f"  {inst.fn_name:8s} {len(df):6d} bars  {df['ts'].min().date()} -> "
              f"{df['ts'].max().date()}  {df['ts'].dt.year.nunique()} yrs "
              f"({n}/280 months)", flush=True)
    print("\ndecade set:", ", ".join(sorted(
        f[:-8] for f in os.listdir(OUT) if f.endswith(".parquet"))))
