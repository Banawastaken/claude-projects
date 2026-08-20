"""Download XAUUSD tick data from Dukascopy and aggregate to M1 bars.

Dukascopy .bi5 format: LZMA-compressed, 20-byte big-endian records of
(ms_offset_in_hour, ask_points, bid_points, ask_volume, bid_volume).
XAUUSD uses a point divisor of 1000 (3 decimals).

Output is one row per minute carrying bid OHLC plus real measured spread
statistics, so the backtester can fill buys at ask and sells at bid using
the spread that actually prevailed in that minute.
"""

import datetime as dt
import lzma
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import random

import numpy as np
import pandas as pd
import requests

_local = threading.local()
FAILED: list = []


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "Mozilla/5.0"
    return _local.s

BASE = "https://datafeed.dukascopy.com/datafeed/XAUUSD"
DIV = 1000.0
CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUT = os.path.join(os.path.dirname(__file__), "..", "data")
REC = struct.Struct(">IIIff")


def hour_url(when: dt.datetime) -> str:
    return f"{BASE}/{when.year:04d}/{when.month - 1:02d}/{when.day:02d}/{when.hour:02d}h_ticks.bi5"


def cache_path(when: dt.datetime) -> str:
    return os.path.join(CACHE, f"{when.year:04d}{when.month:02d}{when.day:02d}{when.hour:02d}.bi5")


def fetch_hour(when: dt.datetime, retries: int = 6):
    """Return (raw_bytes, ok) for one hour, cached on disk. Empty bytes = no ticks.

    A failed download is never cached, so a later pass can pick it up.
    """
    path = cache_path(when)
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read(), True
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = session().get(hour_url(when), timeout=90)
            if resp.status_code == 404:  # no file published for this hour
                raw = b""
            elif resp.status_code in (429, 503, 502, 504):
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
            delay = min(delay * 2, 60.0)
    return b"", False


def decode_hour(raw: bytes, when: dt.datetime):
    """Decode raw bytes to (epoch_ms, bid, ask) arrays."""
    if not raw:
        return None
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError:
        return None
    n = len(data) // 20
    if n == 0:
        return None
    arr = np.frombuffer(data[: n * 20], dtype=">u4").reshape(n, 5)
    ms = arr[:, 0].astype(np.int64)
    ask = arr[:, 1].astype(np.float64) / DIV
    bid = arr[:, 2].astype(np.float64) / DIV
    base = int(when.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    return base + ms, bid, ask


def hour_to_m1(when: dt.datetime) -> pd.DataFrame | None:
    raw, ok = fetch_hour(when)
    if not ok:
        FAILED.append(when)
        return None
    dec = decode_hour(raw, when)
    if dec is None:
        return None
    epoch_ms, bid, ask = dec
    minute = (epoch_ms // 60000).astype(np.int64)
    spread = ask - bid
    df = pd.DataFrame({"minute": minute, "bid": bid, "spread": spread})
    grp = df.groupby("minute", sort=True)
    out = grp.agg(
        open=("bid", "first"),
        high=("bid", "max"),
        low=("bid", "min"),
        close=("bid", "last"),
        spread_mean=("spread", "mean"),
        spread_med=("spread", "median"),
        spread_max=("spread", "max"),
        ticks=("bid", "size"),
    ).reset_index()
    return out


def build(start: dt.date, end: dt.date, workers: int = 8) -> pd.DataFrame:
    hours = []
    cur = dt.datetime(start.year, start.month, start.day, tzinfo=dt.timezone.utc)
    stop = dt.datetime(end.year, end.month, end.day, tzinfo=dt.timezone.utc)
    while cur < stop:
        # Skip hours the gold market is certainly closed (Sat, and Sun before 21:00 UTC).
        wd = cur.weekday()  # Mon=0 .. Sun=6
        if not (wd == 5 or (wd == 6 and cur.hour < 21) or (wd == 4 and cur.hour >= 22)):
            hours.append(cur.replace(tzinfo=None))
        cur += dt.timedelta(hours=1)

    frames = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(hour_to_m1, hours):
            done += 1
            if res is not None:
                frames.append(res)
            if done % 500 == 0:
                print(f"  {done}/{len(hours)} hours, {len(FAILED)} failed", flush=True)

    # Retry whatever the throttle refused, slowly, until it stops improving.
    for attempt in range(6):
        if not FAILED:
            break
        pending, FAILED[:] = list(FAILED), []
        print(f"  retry pass {attempt + 1}: {len(pending)} hours", flush=True)
        time.sleep(20)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for res in pool.map(hour_to_m1, pending):
                if res is not None:
                    frames.append(res)
        if len(FAILED) == len(pending):
            break
    if FAILED:
        print(f"  WARNING: {len(FAILED)} hours unavailable after retries", flush=True)

    df = pd.concat(frames, ignore_index=True).sort_values("minute")
    df = df.drop_duplicates("minute", keep="last").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["minute"] * 60, unit="s", utc=True)
    return df[
        ["ts", "open", "high", "low", "close", "spread_mean", "spread_med", "spread_max", "ticks"]
    ]


if __name__ == "__main__":
    start = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date(2025, 1, 1)
    end = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date(2026, 8, 21)
    print(f"Fetching XAUUSD ticks {start} -> {end}", flush=True)
    df = build(start, end)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "xauusd_m1.parquet")
    df.to_parquet(path, index=False)
    print(f"Wrote {len(df):,} M1 bars to {path}")
    print(df.head())
    print(df.tail())
