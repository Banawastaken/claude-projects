"""Daily bars from Yahoo's chart endpoint.

Used for the asset classes the Dukascopy CFD feed does not carry -- notably
Treasuries and REITs, without which the Faber tactical-allocation replication
would be a different strategy rather than a faithful one.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import pandas as pd

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
CHART = ("https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
         "?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit")
OUT = "data/yahoo"


def fetch(symbol: str, start="2014-01-01", end="2026-12-31", tries=4):
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int(pd.Timestamp(end, tz="UTC").timestamp())
    url = CHART.format(sym=symbol, p1=p1, p2=p2)
    delay = 4.0
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode())
            break
        except Exception:
            time.sleep(delay)
            delay *= 2
    else:
        return None

    res = (d.get("chart") or {}).get("result")
    if not res:
        return None
    res = res[0]
    q = res["indicators"]["quote"][0]
    adj = (res["indicators"].get("adjclose") or [{}])[0].get("adjclose")
    df = pd.DataFrame({
        "ts": pd.to_datetime(res["timestamp"], unit="s", utc=True),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "volume": q["volume"],
    })
    # Total return matters for anything held for months, so the adjusted close
    # is what the strategies actually trade on.
    df["adj_close"] = adj if adj is not None else df["close"]
    df = df.dropna(subset=["close", "adj_close"]).reset_index(drop=True)
    df["date"] = df["ts"].dt.tz_convert("America/New_York").dt.normalize()
    return df


def build(symbols, out=OUT, gap=3.0):
    os.makedirs(out, exist_ok=True)
    done = []
    for i, s in enumerate(symbols):
        path = os.path.join(out, f"{s}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            print(f"  {s:6s} cached  {len(df):>5,} rows", flush=True)
            done.append(s)
            continue
        if i:
            time.sleep(gap)
        df = fetch(s)
        if df is None or len(df) < 500:
            print(f"  {s:6s} FAILED", flush=True)
            continue
        df.to_parquet(path, index=False)
        print(f"  {s:6s} {len(df):>5,} rows  {df['date'].min().date()} .. "
              f"{df['date'].max().date()}", flush=True)
        done.append(s)
    return done


if __name__ == "__main__":
    syms = sys.argv[1:] or ["SPY", "EFA", "IEF", "VNQ", "DBC"]
    build(syms)
