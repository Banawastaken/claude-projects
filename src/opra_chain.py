"""Rebuild dated NDX option chains from OPRA, and compute dealer gamma on them.

Three files meet here. `definition` says what each contract is, `statistics`
says how much open interest it carried at each day's close, and `ohlcv-1d`
gives the settlement price that implied volatility is inverted from.

Gamma is then computed with the same Black-Scholes call `gex.py` uses on the
live chain, so a level measured in 2025 means the same thing as one measured
today. That is the point of doing it this way rather than buying a vendor's
published levels: the historical series and the live recorder are one
measurement, not two that happen to share a name.
"""

from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gex as G  # noqa: E402

OUT = "data/opra"
PRICE_SCALE = 1e9
# OPRA statistic type for end-of-day open interest.
STAT_OPEN_INTEREST = 9


def _files(schema, root=OUT):
    return sorted(glob.glob(os.path.join(root, f"NDX_OPT_{schema}_*.dbn.zst")))


def definitions(root=OUT, cache=os.path.join(OUT, "definitions.parquet")):
    """instrument_id -> strike, expiry, right, for every contract seen."""
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    import databento as db
    rows = {}
    for f in _files("definition", root):
        for r in db.DBNStore.from_file(f):
            iid = r.instrument_id
            if iid in rows:
                continue
            try:
                cp = r.instrument_class
                strike = r.strike_price / PRICE_SCALE
                exp = pd.Timestamp(r.expiration, unit="ns", tz="UTC")
            except Exception:
                continue
            if strike <= 0:
                continue
            rows[iid] = {"instrument_id": iid, "strike": strike,
                         "expiry": exp,
                         "right": 1 if str(cp).upper().startswith("C") else -1,
                         "raw_symbol": getattr(r, "raw_symbol", "")}
        print(f"  {os.path.basename(f)}: {len(rows):,} contracts so far", flush=True)
    df = pd.DataFrame(rows.values())
    df.to_parquet(cache, index=False)
    return df


def open_interest(root=OUT, cache=os.path.join(OUT, "open_interest.parquet")):
    """date, instrument_id, open interest at that day's close."""
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    import databento as db
    recs = []
    for f in _files("statistics", root):
        n = 0
        for r in db.DBNStore.from_file(f):
            if r.stat_type != STAT_OPEN_INTEREST:
                continue
            recs.append((r.ts_ref, r.instrument_id, r.quantity))
            n += 1
        print(f"  {os.path.basename(f)}: {n:,} OI records", flush=True)
    df = pd.DataFrame(recs, columns=["ts", "instrument_id", "oi"])
    df["date"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.normalize()
    df = df.drop(columns=["ts"])
    df = df[df["oi"] > 0].drop_duplicates(["date", "instrument_id"], keep="last")
    df.to_parquet(cache, index=False)
    return df


def settles(root=OUT, cache=os.path.join(OUT, "settles.parquet")):
    """date, instrument_id, closing price."""
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    import databento as db
    recs = []
    for f in _files("ohlcv-1d", root):
        for r in db.DBNStore.from_file(f):
            if r.close > 0:
                recs.append((r.ts_event, r.instrument_id, r.close / PRICE_SCALE))
    df = pd.DataFrame(recs, columns=["ts", "instrument_id", "close"])
    df["date"] = pd.to_datetime(df["ts"], unit="ns", utc=True).dt.normalize()
    df = df.drop(columns=["ts"]).drop_duplicates(["date", "instrument_id"],
                                                 keep="last")
    df.to_parquet(cache, index=False)
    return df


def implied_vol(price, spot, strike, t, right, r=0.04, lo=0.01, hi=3.0):
    """Invert Black-Scholes for volatility by bisection.

    Bisection rather than Newton: it cannot diverge on the deep wings where
    vega is nearly zero, which is exactly where a gamma profile needs the
    contracts to still be present rather than dropped.
    """
    t = max(float(t), 1.0 / 365.0)
    def bs(sig):
        from math import erf, exp, log, sqrt
        d1 = (log(spot / strike) + (r + 0.5 * sig ** 2) * t) / (sig * sqrt(t))
        d2 = d1 - sig * sqrt(t)
        N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
        if right > 0:
            return spot * N(d1) - strike * exp(-r * t) * N(d2)
        return strike * exp(-r * t) * N(-d2) - spot * N(-d1)

    a, b = lo, hi
    fa = bs(a) - price
    fb = bs(b) - price
    if fa * fb > 0:
        return np.nan
    for _ in range(60):
        m = 0.5 * (a + b)
        fm = bs(m) - price
        if abs(fm) < 1e-8:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)
