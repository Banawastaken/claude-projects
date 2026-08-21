"""Fetch a decade of XAUUSD H1 bid/ask in one go.

`fetch_h1.build_instrument` walks months serially, which is fine when the
parallelism sits across instruments but painfully slow for a single symbol over
eleven years. This prefetches every monthly file concurrently first, then lets
the normal builder assemble them from cache.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_h1 as F  # noqa: E402
from universe import BY_FN  # noqa: E402

START = dt.date(2015, 1, 1)
END = dt.date(2026, 8, 1)
OUT_NAME = "XAUUSD_H1_DECADE"

if __name__ == "__main__":
    inst = BY_FN["XAUUSD"]
    ms = F.months(START, END)
    jobs = [(y, m, side) for (y, m) in ms for side in ("BID", "ASK")]
    print(f"prefetching {len(jobs)} monthly files ({START} -> {END})", flush=True)

    def go(j):
        y, m, side = j
        return F.fetch_month(inst.duka, y, m, side)

    with ThreadPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(go, jobs))
    print(f"fetched ok: {sum(1 for r in res if r[1])}/{len(res)}", flush=True)

    df = F.build_instrument(inst, START, END)
    if df is None:
        print("FAILED to assemble")
        sys.exit(1)
    out = os.path.join(F.OUT, f"{OUT_NAME}.parquet")
    os.makedirs(F.OUT, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df):,} bars  {df['ts'].min().date()} -> {df['ts'].max().date()}  "
          f"divisor {df.attrs['divisor']:.0f}")
    print(df.groupby(df["ts"].dt.year).size().to_string())
