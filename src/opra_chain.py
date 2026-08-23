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
    """date, instrument_id, open interest known at that morning's open.

    Two details decide whether this is usable.

    The timestamp is `ts_event`, not `ts_ref`: OPRA leaves `ts_ref` as the
    uint64 sentinel on open-interest records, and reading it collapsed a year
    of data onto a single date without erroring.

    Every contract's figure arrives from four publishers at 10:30 UTC, before
    the cash open, and they agree exactly -- checked across a month, zero
    disagreements -- so the duplicates are one OCC number disseminated four
    times and must be deduplicated rather than summed. Summing would have
    quadrupled every gamma level.

    Stamped pre-open, the figure describes the previous close and is therefore
    known before the session it is keyed to: what dealers are carrying into
    that day.
    """
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    import databento as db
    recs = []
    for f in _files("statistics", root):
        n = 0
        for r in db.DBNStore.from_file(f):
            if r.stat_type != STAT_OPEN_INTEREST:
                continue
            recs.append((r.ts_event, r.instrument_id, r.quantity))
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


def implied_vol_vec(price, spot, strike, t, right, r=0.04, lo=0.005, hi=4.0,
                    iters=60):
    """Vectorised Black-Scholes inversion by bisection over whole chains.

    A year of NDX is 578,000 contract-days; inverting them one at a time in
    Python is tens of minutes, and the same bisection across numpy arrays is
    seconds. Bisection rather than Newton for the same reason as before -- it
    cannot diverge on the deep wings where vega is nearly zero, and those are
    exactly the strikes a gamma profile must keep.
    """
    from scipy.special import ndtr

    price = np.asarray(price, float)
    spot = np.asarray(spot, float)
    strike = np.asarray(strike, float)
    t = np.maximum(np.asarray(t, float), 1.0 / 365.0)
    right = np.asarray(right, float)

    def bs(sig):
        vt = sig * np.sqrt(t)
        d1 = (np.log(spot / strike) + (r + 0.5 * sig ** 2) * t) / vt
        d2 = d1 - vt
        disc = strike * np.exp(-r * t)
        call = spot * ndtr(d1) - disc * ndtr(d2)
        put = disc * ndtr(-d2) - spot * ndtr(-d1)
        return np.where(right > 0, call, put)

    a = np.full_like(price, lo)
    b = np.full_like(price, hi)
    fa = bs(a) - price
    fb = bs(b) - price
    bracketed = (fa * fb) <= 0
    for _ in range(iters):
        m = 0.5 * (a + b)
        fm = bs(m) - price
        left = (fa * fm) < 0
        b = np.where(left, m, b)
        a = np.where(left, a, m)
        fa = np.where(left, fa, fm)
    return np.where(bracketed, 0.5 * (a + b), np.nan)


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
