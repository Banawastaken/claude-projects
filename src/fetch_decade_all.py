"""Build the research dataset: every instrument, every year from 2015.

The divisor is detected once per symbol against a 2025 month, where the
configured price range is known to be right, and then reused for the whole
history. Detecting it per-year would fail on the early years -- gold traded
near $1,050 in 2015 and Bitcoin near $300, both far outside any range that
also has to describe 2026.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_h1 as F  # noqa: E402
from universe import UNIVERSE, Instrument  # noqa: E402

START = dt.date(2015, 1, 1)
END = dt.date(2026, 8, 1)
REF = (2026, 6)  # a month whose prices sit inside the configured range
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "decade")


def divisor_for(inst: Instrument):
    raw, ok = F.fetch_month(inst.duka, REF[0], REF[1], "BID")
    if not ok or not raw:
        return None
    return F.detect_divisor(raw, REF[0], REF[1], inst.price_range)


def build(inst: Instrument, divisor: float):
    frames = []
    for (y, m) in F.months(START, END):
        braw, bok = F.fetch_month(inst.duka, y, m, "BID")
        araw, aok = F.fetch_month(inst.duka, y, m, "ASK")
        if not (bok and aok) or not braw or not araw:
            continue
        b = F.decode_month(braw, y, m, divisor)
        a = F.decode_month(araw, y, m, divisor)
        if b is None or a is None:
            continue
        mg = b.merge(a, on="ts_epoch", suffixes=("", "_a"))
        if mg.empty:
            continue
        frames.append(pd.DataFrame({
            "ts_epoch": mg["ts_epoch"],
            "open": mg["o"], "high": mg["h"], "low": mg["l"], "close": mg["c"],
            "ask_open": mg["o_a"], "ask_high": mg["h_a"],
            "ask_low": mg["l_a"], "ask_close": mg["c_a"],
            "volume": mg["v"] + mg["v_a"],
        }))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True).sort_values("ts_epoch")
    df = df.drop_duplicates("ts_epoch", keep="last").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts_epoch"], unit="s", utc=True)
    df["minute"] = df["ts_epoch"] // 60
    df["spread_med"] = df["ask_close"] - df["close"]
    rng = df["high"] - df["low"]
    dead = (df["volume"] == 0) & (rng == 0)
    bad = (df["spread_med"] < 0) | (df["spread_med"] > df["close"].abs() * 0.05)
    return df[~dead & ~bad].reset_index(drop=True)


def one(inst: Instrument):
    div = divisor_for(inst)
    if div is None:
        return inst.fn_name, None, "divisor not detected"
    jobs = [(y, m, s) for (y, m) in F.months(START, END) for s in ("BID", "ASK")]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda j: F.fetch_month(inst.duka, j[0], j[1], j[2]), jobs))
    df = build(inst, div)
    if df is None or len(df) < 5000:
        return inst.fn_name, None, "insufficient history"
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, f"{inst.fn_name}.parquet"), index=False)
    years = df["ts"].dt.year.nunique()
    return inst.fn_name, len(df), (f"{df['ts'].min().date()} -> {df['ts'].max().date()}"
                                   f"  {years} yrs  div {div:.0f}")


if __name__ == "__main__":
    print(f"Building decade dataset for {len(UNIVERSE)} instruments\n", flush=True)
    for inst in UNIVERSE:
        name, n, info = one(inst)
        if n is None:
            print(f"  {name:8s} SKIP  {info}", flush=True)
        else:
            print(f"  {name:8s} {n:6d} bars  {info}", flush=True)
