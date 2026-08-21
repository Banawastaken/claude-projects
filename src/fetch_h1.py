"""Download monthly H1 BID/ASK candles for the whole FundedNext universe.

Dukascopy publishes a month of hourly candles in a single file, which is what
makes a thirty-instrument study affordable: 20 months x 2 sides = 40 requests
per instrument instead of the ~1,000 a minute-level download would need.

Price scaling differs per symbol -- five-decimal FX divides by 100,000 while
gold and the index CFDs divide by 1,000 -- so the divisor is detected by
decoding with each candidate and keeping the one that lands the median price
inside the instrument's known trading range.
"""

from __future__ import annotations

import datetime as dt
import lzma
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from universe import UNIVERSE, Instrument  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "data", "h1")
OUT = os.path.join(HERE, "..", "data", "instruments")
BASE = "https://datafeed.dukascopy.com/datafeed"
DIVISORS = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]

_local = threading.local()
_gate = threading.Semaphore(10)
_cool = [0.0]
_lock = threading.Lock()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "Mozilla/5.0"
    return _local.s


def _wait():
    while True:
        with _lock:
            left = _cool[0] - time.time()
        if left <= 0:
            return
        time.sleep(min(left, 5.0) + random.random())


def fetch_month(sym: str, year: int, month: int, side: str, retries: int = 6):
    path = os.path.join(CACHE, sym, f"{year:04d}{month:02d}_{side}.bi5")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return fh.read(), True
    url = f"{BASE}/{sym}/{year:04d}/{month - 1:02d}/{side}_candles_hour_1.bi5"
    delay = 3.0
    for attempt in range(retries):
        _wait()
        try:
            with _gate:
                r = session().get(url, timeout=120)
            if r.status_code == 404:
                raw = b""
            elif r.status_code in (429, 500, 502, 503, 504):
                with _lock:
                    _cool[0] = max(_cool[0], time.time() + 8.0)
                raise RuntimeError(str(r.status_code))
            else:
                r.raise_for_status()
                raw = r.content
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


def decode_month(raw: bytes, year: int, month: int, divisor: float):
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
    vol = np.frombuffer(buf, dtype=">f4").reshape(n, 6)[:, 5].astype(np.float64)
    secs = ints[:, 0].astype(np.int64)
    o = ints[:, 1] / divisor
    c = ints[:, 2] / divisor
    lo = ints[:, 3] / divisor
    hi = ints[:, 4] / divisor
    valid = (o > 0) & (c > 0) & (hi > 0) & (lo > 0)
    if not valid.any():
        return None
    base = int(dt.datetime(year, month, 1, tzinfo=dt.timezone.utc).timestamp())
    return pd.DataFrame({
        "ts_epoch": base + secs[valid],
        "o": o[valid], "h": hi[valid], "l": lo[valid], "c": c[valid],
        "v": vol[valid],
    })


def detect_divisor(raw: bytes, year: int, month: int, rng: tuple[float, float]) -> float | None:
    for div in DIVISORS:
        df = decode_month(raw, year, month, div)
        if df is None or df.empty:
            continue
        med = float(df["c"].median())
        if rng[0] <= med <= rng[1]:
            return div
    return None


def months(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    out = []
    while (y, m) <= (end.year, end.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def build_instrument(inst: Instrument, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    ms = months(start, end)
    divisor = None
    frames = []
    for (y, m) in ms:
        braw, bok = fetch_month(inst.duka, y, m, "BID")
        araw, aok = fetch_month(inst.duka, y, m, "ASK")
        if not (bok and aok) or not braw or not araw:
            continue
        if divisor is None:
            divisor = detect_divisor(braw, y, m, inst.price_range)
            if divisor is None:
                continue
        b = decode_month(braw, y, m, divisor)
        a = decode_month(araw, y, m, divisor)
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
    if not frames or divisor is None:
        return None
    df = pd.concat(frames, ignore_index=True).sort_values("ts_epoch")
    df = df.drop_duplicates("ts_epoch", keep="last").reset_index(drop=True)
    df["ts"] = pd.to_datetime(df["ts_epoch"], unit="s", utc=True)
    df["minute"] = df["ts_epoch"] // 60
    df["spread_med"] = df["ask_close"] - df["close"]

    # Drop hours the market was shut (no ticks) and any crossed quote.
    rng = df["high"] - df["low"]
    dead = (df["volume"] == 0) & (rng == 0)
    bad = (df["spread_med"] < 0) | (df["spread_med"] > df["close"].abs() * 0.05)
    df = df[~dead & ~bad].reset_index(drop=True)
    df.attrs["divisor"] = divisor
    return df


def one(inst: Instrument):
    try:
        df = build_instrument(inst, dt.date(2025, 1, 1), dt.date(2026, 8, 1))
    except Exception as exc:
        return inst.fn_name, None, f"error {exc}"
    if df is None or len(df) < 2000:
        return inst.fn_name, None, "no usable data"
    os.makedirs(OUT, exist_ok=True)
    df.to_parquet(os.path.join(OUT, f"{inst.fn_name}.parquet"), index=False)
    med_spread = float(df["spread_med"].median())
    return inst.fn_name, len(df), (
        f"{df['ts'].min().date()} -> {df['ts'].max().date()}  "
        f"px {df['close'].median():.4f}  spread {med_spread:.5f}  "
        f"div {df.attrs['divisor']:.0f}")


if __name__ == "__main__":
    print(f"Fetching H1 bid/ask for {len(UNIVERSE)} instruments\n")
    with ThreadPoolExecutor(max_workers=10) as pool:
        for name, n, info in pool.map(one, UNIVERSE):
            if n is None:
                print(f"  {name:8s} FAILED  {info}", flush=True)
            else:
                print(f"  {name:8s} {n:6d} bars  {info}", flush=True)
