"""Find which Dukascopy instruments have usable M1 candle data.

Probes one known-good trading day per symbol and reports whether both the BID
and ASK candle files exist and decode.
"""

import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fetch_candles as fc  # noqa: E402

CANDIDATES = [
    # majors / minors
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "CADJPY", "CHFJPY", "EURAUD",
    # metals
    "XAUUSD", "XAGUSD", "XPTUSD", "XPDUSD",
    # indices
    "USA500IDXUSD", "USATECHIDXUSD", "USA30IDXUSD", "USSC2000IDXUSD",
    "DEUIDXEUR", "GBRIDXGBP", "JPNIDXJPY", "FRAIDXEUR", "AUSIDXAUD",
    "HKGIDXHKD", "EUSIDXEUR", "ESPIDXEUR",
    # energy / softs
    "LIGHTCMDUSD", "BRENTCMDUSD", "NATGASCMDUSD", "COPPERCMDUSD",
    # crypto
    "BTCUSD", "ETHUSD",
]

PROBE_DAY = dt.date(2026, 6, 11)  # a normal Thursday with known gold data


def probe(sym: str):
    old = fc.BASE
    fc.BASE = f"https://datafeed.dukascopy.com/datafeed/{sym}"
    try:
        ok = {}
        for side in ("BID", "ASK"):
            raw, got = fc.fetch(PROBE_DAY, side, retries=3)
            if not got or not raw:
                ok[side] = 0
                continue
            df = fc.decode(raw, PROBE_DAY)
            ok[side] = 0 if df is None else len(df)
        return ok
    finally:
        fc.BASE = old


if __name__ == "__main__":
    print(f"Probing {len(CANDIDATES)} symbols on {PROBE_DAY}\n")
    good, bad = [], []
    for sym in CANDIDATES:
        # cache under a per-symbol directory so probes do not collide
        fc.CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "data", "candles", sym)
        try:
            r = probe(sym)
        except Exception as exc:
            print(f"  {sym:16s} ERROR {exc}")
            bad.append(sym)
            continue
        if r["BID"] > 100 and r["ASK"] > 100:
            print(f"  {sym:16s} OK   bid={r['BID']:4d} ask={r['ASK']:4d} bars")
            good.append(sym)
        else:
            print(f"  {sym:16s} no data ({r})")
            bad.append(sym)
        time.sleep(1.0)
    print(f"\nusable: {len(good)}")
    print(" ".join(good))
    print(f"\nunavailable: {' '.join(bad)}")
