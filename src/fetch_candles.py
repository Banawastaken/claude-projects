"""Download XAUUSD M1 BID and ASK candles from Dukascopy.

One request per day per side instead of 24 tick files per day, which matters
because the upstream link is slow and Dukascopy throttles aggressively.

Daily candle file: `/{INSTR}/{yyyy}/{MM-1}/{dd}/{BID|ASK}_candles_min_1.bi5`,
LZMA-compressed, 1440 records of 24 bytes:
    >5if = (seconds_from_midnight, open, close, low, high, volume)
with the four prices as integers scaled by 1000 for XAUUSD.

Having both sides gives the true bid/ask spread per minute, so fills can be
taken on the correct side of the book rather than from a spread assumption.
"""

from __future__ import annotations

import datetime as dt
import lzma
import os
import random
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

BASE = "https://datafeed.dukascopy.com/datafeed/XAUUSD"
DIV = 1000.0
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "data", "candles")
OUT = os.path.join(HERE, "..", "data")

_local = threading.local()
_gate = threading.Semaphore(10)
_cooldown_until = [0.0]
_cd_lock = threading.Lock()
FAILED: list = []


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "Mozilla/5.0"
    return _local.s


def cooldown(seconds: float):
    with _cd_lock:
        _cooldown_until[0] = max(_cooldown_until[0], time.time() + seconds)


def wait_turn():
    while True:
        with _cd_lock:
            remaining = _cooldown_until[0] - time.time()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 5.0) + random.random())


def url_for(day: dt.date, side: str) -> str:
    return f"{BASE}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{side}_candles_min_1.bi5"


def cache_for(day: dt.date, side: str) -> str:
    return os.path.join(CACHE, f"{day.isoformat()}_{side}.bi5")


def fetch(day: dt.date, side: str, retries: int = 8):
    path = cache_for(day, side)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read(), True
    delay = 3.0
    for attempt in range(retries):
        wait_turn()
        try:
            with _gate:
                resp = session().get(url_for(day, side), timeout=120)
            if resp.status_code == 404:
                raw = b""
            elif resp.status_code in (429, 500, 502, 503, 504):
                cooldown(8.0)
                raise RuntimeError(f"throttled {resp.status_code}")
            else:
                resp.raise_for_status()
                raw = resp.content
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(raw)
            os.replace(tmp, path)
            return raw, True
        except Exception:
            if attempt == retries - 1:
                return b"", False
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 1.8, 90.0)
    return b"", False


def decode(raw: bytes, day: dt.date):
    """Return DataFrame of minute, open, high, low, close for one side."""
    if not raw:
        return None
    try:
        d = lzma.decompress(raw)
    except lzma.LZMAError:
        return None
    n = len(d) // 24
    if n == 0:
        return None
    buf = d[: n * 24]
    ints = np.frombuffer(buf, dtype=">i4").reshape(n, 6)
    secs = ints[:, 0].astype(np.int64)
    o = ints[:, 1].astype(np.float64) / DIV
    c = ints[:, 2].astype(np.float64) / DIV
    lo = ints[:, 3].astype(np.float64) / DIV
    hi = ints[:, 4].astype(np.float64) / DIV
    vol = np.frombuffer(buf, dtype=">f4").reshape(n, 6)[:, 5].astype(np.float64)

    # minutes with no ticks are published as zero-price filler
    valid = (o > 0) & (c > 0) & (hi > 0) & (lo > 0)
    if not valid.any():
        return None
    base = int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp())
    minute = (base + secs[valid]) // 60
    return pd.DataFrame(
        {
            "minute": minute,
            "o": o[valid],
            "h": hi[valid],
            "l": lo[valid],
            "c": c[valid],
            "v": vol[valid],
        }
    )


def day_frame(day: dt.date):
    out = {}
    for side in ("BID", "ASK"):
        raw, ok = fetch(day, side)
        if not ok:
            FAILED.append(day)
            return None
        df = decode(raw, day)
        if df is None:
            return None
        out[side] = df
    b, a = out["BID"], out["ASK"]
    m = b.merge(a, on="minute", suffixes=("", "_a"))
    if m.empty:
        return None
    return pd.DataFrame(
        {
            "minute": m["minute"],
            "open": m["o"],
            "high": m["h"],
            "low": m["l"],
            "close": m["c"],
            "ask_open": m["o_a"],
            "ask_high": m["h_a"],
            "ask_low": m["l_a"],
            "ask_close": m["c_a"],
            "volume": m["v"] + m["v_a"],
        }
    )


def build(start: dt.date, end: dt.date, workers: int = 10) -> pd.DataFrame:
    days = []
    d = start
    while d < end:
        if d.weekday() != 5:  # Saturdays are never published
            days.append(d)
        d += dt.timedelta(days=1)

    frames = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(day_frame, days):
            done += 1
            if res is not None:
                frames.append(res)
            if done % 25 == 0:
                print(f"  {done}/{len(days)} days, {len(FAILED)} failed", flush=True)

    for attempt in range(8):
        if not FAILED:
            break
        pending = sorted(set(FAILED))
        FAILED[:] = []
        print(f"  retry pass {attempt + 1}: {len(pending)} days", flush=True)
        time.sleep(30)
        with ThreadPoolExecutor(max_workers=3) as pool:
            for res in pool.map(day_frame, pending):
                if res is not None:
                    frames.append(res)
        if len(set(FAILED)) >= len(pending):
            break
    if FAILED:
        print(f"  WARNING: {len(set(FAILED))} days unavailable", flush=True)

    df = pd.concat(frames, ignore_index=True).sort_values("minute")
    df = df.drop_duplicates("minute", keep="last").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["minute"] * 60, unit="s", utc=True)
    df["spread_med"] = df["ask_close"] - df["close"]
    # a handful of published minutes carry crossed or absurd quotes
    bad = (df["spread_med"] < 0) | (df["spread_med"] > 15)
    if bad.any():
        print(f"  dropping {int(bad.sum())} bars with implausible spread")
        df = df[~bad].reset_index(drop=True)
    return df


if __name__ == "__main__":
    start = dt.date.fromisoformat(sys.argv[1])
    end = dt.date.fromisoformat(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else "xauusd_m1"
    print(f"Fetching XAUUSD M1 candles {start} -> {end}", flush=True)
    df = build(start, end)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tag}.parquet")
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df):,} bars -> {path}")
    print(df[["ts", "open", "high", "low", "close", "spread_med"]].head(3).to_string(index=False))
    print(df[["ts", "open", "high", "low", "close", "spread_med"]].tail(3).to_string(index=False))
    print("median spread:", df["spread_med"].median().round(4))
