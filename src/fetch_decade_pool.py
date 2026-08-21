"""Decade fetch with one global job pool instead of instrument-by-instrument.

Fetching per instrument serialises 280 requests behind a link that takes about
ten seconds each, so the whole universe would take hours. Every
(instrument, month, side) job goes into a single pool instead, and a priority
subset is fetched first so screening can start while the rest arrives.

Usage:  fetch_decade_pool.py [priority|rest|all]
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_h1 as F  # noqa: E402
from fetch_decade_all import START, END, build, divisor_for  # noqa: E402
from universe import UNIVERSE  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "decade")

# A spread of asset classes, enough to judge breadth without the whole list.
PRIORITY = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "SPX500", "NDX100", "GER30", "US30", "USOUSD", "XAGUSD"]


def pick(which):
    if which == "priority":
        return [i for i in UNIVERSE if i.fn_name in PRIORITY]
    if which == "rest":
        return [i for i in UNIVERSE if i.fn_name not in PRIORITY]
    return list(UNIVERSE)


def main(which):
    insts = pick(which)
    print(f"decade fetch: {len(insts)} instruments ({which})", flush=True)

    jobs = []
    for inst in insts:
        for (y, m) in F.months(START, END):
            for side in ("BID", "ASK"):
                jobs.append((inst.duka, y, m, side))
    todo = [j for j in jobs
            if not os.path.exists(os.path.join(F.CACHE, j[0], f"{j[1]:04d}{j[2]:02d}_{j[3]}.bi5"))]
    print(f"{len(jobs)} files, {len(todo)} to fetch", flush=True)

    done = [0]

    def go(j):
        r = F.fetch_month(j[0], j[1], j[2], j[3])
        done[0] += 1
        if done[0] % 250 == 0:
            print(f"  {done[0]}/{len(todo)}", flush=True)
        return r

    if todo:
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(go, todo))

    os.makedirs(OUT, exist_ok=True)
    for inst in insts:
        path = os.path.join(OUT, f"{inst.fn_name}.parquet")
        if os.path.exists(path):
            continue
        div = divisor_for(inst)
        if div is None:
            print(f"  {inst.fn_name:8s} SKIP (divisor)", flush=True)
            continue
        df = build(inst, div)
        if df is None or len(df) < 5000:
            print(f"  {inst.fn_name:8s} SKIP (insufficient)", flush=True)
            continue
        df.to_parquet(path, index=False)
        print(f"  {inst.fn_name:8s} {len(df):6d} bars  "
              f"{df['ts'].min().date()} -> {df['ts'].max().date()}  "
              f"{df['ts'].dt.year.nunique()} yrs", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "priority")
