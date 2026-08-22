"""Dealer gamma exposure from a CBOE delayed option chain.

Everything here is computed from the chain itself rather than taken from a
vendor's published levels, so the numbers can be checked line by line.

The dealer-positioning convention is the one everyone else publishes, from the
original SqueezeMetrics note onward: dealers are assumed long calls and short
puts, so GEX = call gamma x OI - put gamma x OI.  Positive total GEX then means
dealers hedge against the move (selling strength, buying weakness) and negative
GEX means they hedge with it.

That sign is an assumption about who holds what, not a measurement, and it is
the assumption every "trade the gamma flip" rule inherits.  `dealer_sign` makes
it a parameter so the opposite convention can be tested rather than argued
about.

Gamma is re-priced with Black-Scholes at every candidate spot rather than held
at its quoted value.  That matters: the gamma flip is defined as the spot where
the profile crosses zero, and a profile built from frozen per-strike gammas is
not the same curve at all.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import urllib.request

import numpy as np

CBOE = "https://cdn.cboe.com/api/global/delayed_quotes/options/{sym}.json"
CONTRACT_MULT = 100.0

# OSI: root, yymmdd, C/P, strike in thousandths padded to 8 digits.
_OSI = re.compile(r"^([A-Z^]+)(\d{6})([CP])(\d{8})$")


def fetch_chain(symbol: str = "_NDX", path: str | None = None) -> dict:
    """Return CBOE's raw payload, from `path` if given else over the network."""
    if path:
        with open(path) as fh:
            return json.load(fh)
    req = urllib.request.Request(
        CBOE.format(sym=symbol),
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read().decode())


def parse_chain(payload: dict, asof: dt.date | None = None):
    """Flatten the payload into arrays, dropping rows that cannot carry gamma.

    Returns (spot, dict-of-arrays). Contracts with no open interest, no IV or
    already expired contribute nothing to dealer gamma and are removed here so
    every later step can assume the arrays are clean.
    """
    data = payload["data"]
    spot = float(data["current_price"])
    ts = payload.get("timestamp", "")
    if asof is None:
        asof = (dt.datetime.fromisoformat(ts.split(" GMT")[0].strip()).date()
                if ts else dt.date.today())

    strike, expiry, right, oi, iv, gam, vol = [], [], [], [], [], [], []
    for o in data["options"]:
        m = _OSI.match(o["option"])
        if not m:
            continue
        _, ymd, cp, k = m.groups()
        exp = dt.date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
        strike.append(int(k) / 1000.0)
        expiry.append((exp - asof).days)
        right.append(1 if cp == "C" else -1)
        oi.append(float(o.get("open_interest") or 0.0))
        iv.append(float(o.get("iv") or 0.0))
        gam.append(float(o.get("gamma") or 0.0))
        vol.append(float(o.get("volume") or 0.0))

    a = {k: np.asarray(v, dtype=float) for k, v in
         dict(strike=strike, dte=expiry, right=right, oi=oi, iv=iv,
              gamma_quoted=gam, volume=vol).items()}
    keep = (a["oi"] > 0) & (a["iv"] > 0) & (a["dte"] >= 0)
    return spot, {k: v[keep] for k, v in a.items()}


def bs_gamma(spot, strike, dte_days, iv, r=0.04, q=0.0):
    """Black-Scholes gamma, vectorised over contracts and safe at expiry.

    Same-day expiries are floored at a few minutes of calendar time so 0DTE
    contracts produce a large-but-finite gamma spike instead of a divide by
    zero.
    """
    t = np.maximum(np.asarray(dte_days, dtype=float), 0.0) / 365.0
    t = np.maximum(t, 1.0 / (365.0 * 24.0 * 6.0))
    s = np.asarray(spot, dtype=float)
    sig = np.maximum(np.asarray(iv, dtype=float), 1e-6)
    vt = sig * np.sqrt(t)
    d1 = (np.log(s / strike) + (r - q + 0.5 * sig ** 2) * t) / vt
    return np.exp(-q * t) * np.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi) / (s * vt)


def gex_at(spot, ch, dealer_sign=(1.0, -1.0), r=0.04, q=0.0, max_dte=None):
    """Dollar gamma per 1% move, per contract, at a hypothetical `spot`.

    `dealer_sign` is (call, put): dealers long calls, short puts by default.
    """
    m = np.ones(len(ch["strike"]), dtype=bool) if max_dte is None else ch["dte"] <= max_dte
    g = bs_gamma(spot, ch["strike"][m], ch["dte"][m], ch["iv"][m], r=r, q=q)
    sign = np.where(ch["right"][m] > 0, dealer_sign[0], dealer_sign[1])
    return sign * g * ch["oi"][m] * CONTRACT_MULT * spot ** 2 * 0.01, m


def profile(ch, spot, lo=0.90, hi=1.10, n=201, **kw):
    """Total dealer GEX across a grid of hypothetical spots."""
    grid = np.linspace(spot * lo, spot * hi, n)
    tot = np.array([gex_at(s, ch, **kw)[0].sum() for s in grid])
    return grid, tot


def gamma_flip(grid, tot):
    """Spot where the gamma profile crosses zero, nearest to where it changes sign.

    Returns None when the profile never changes sign over the grid -- a real
    state (all-positive or all-negative gamma), not an error.
    """
    s = np.sign(tot)
    idx = np.where(np.diff(s) != 0)[0]
    if len(idx) == 0:
        return None
    # The flip that matters is the last crossing: below it dealers are short
    # gamma, above it long.
    i = idx[-1]
    x0, x1, y0, y1 = grid[i], grid[i + 1], tot[i], tot[i + 1]
    if y1 == y0:
        return float(x0)
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def by_strike(ch, spot, **kw):
    """Per-strike dealer GEX at the current spot, aggregated over expiries."""
    vals, m = gex_at(spot, ch, **kw)
    ks = ch["strike"][m]
    uniq = np.unique(ks)
    agg = np.array([vals[ks == k].sum() for k in uniq])
    return uniq, agg


def walls(strikes, agg, spot, width=0.10):
    """Largest positive (call) and most negative (put) gamma strikes near spot."""
    near = (strikes > spot * (1 - width)) & (strikes < spot * (1 + width))
    if not near.any():
        return None, None
    k, v = strikes[near], agg[near]
    return float(k[np.argmax(v)]), float(k[np.argmin(v)])


def snapshot(symbol="_NDX", path=None, max_dte=None, **kw):
    """One complete GEX reading: the row we can store and later backtest on."""
    payload = fetch_chain(symbol, path)
    spot, ch = parse_chain(payload)
    grid, tot = profile(ch, spot, max_dte=max_dte, **kw)
    total = float(gex_at(spot, ch, max_dte=max_dte, **kw)[0].sum())
    flip = gamma_flip(grid, tot)
    ks, agg = by_strike(ch, spot, max_dte=max_dte, **kw)
    cw, pw = walls(ks, agg, spot)
    used = np.ones(len(ch["strike"]), dtype=bool) if max_dte is None else ch["dte"] <= max_dte
    return {
        "symbol": payload.get("symbol", symbol),
        "timestamp": payload.get("timestamp", ""),
        "spot": spot,
        "total_gex": total,
        "gamma_flip": flip,
        "flip_distance_pct": None if flip is None else (spot / flip - 1.0) * 100.0,
        "call_wall": cw,
        "put_wall": pw,
        "n_contracts": int(used.sum()),
        "total_oi": float(ch["oi"][used].sum()),
        "max_dte": max_dte,
    }


def record(out_dir, symbols=("_NDX", "_SPX"), max_dte=None):
    """Append today's snapshot for each symbol to a per-symbol JSONL history.

    Free chains are a live snapshot only, so the only way to obtain history
    here is to start keeping it.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sym in symbols:
        try:
            snap = snapshot(sym, max_dte=max_dte)
        except Exception as e:  # a single bad symbol should not lose the rest
            rows.append({"symbol": sym, "error": str(e)})
            continue
        with open(os.path.join(out_dir, f"{sym.strip('_')}.jsonl"), "a") as fh:
            fh.write(json.dumps(snap) + "\n")
        rows.append(snap)
    return rows


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "_NDX"
    src = sys.argv[2] if len(sys.argv) > 2 else None
    for dte in (None, 0, 5):
        snap = snapshot(sym, src, max_dte=dte)
        tag = "all expiries" if dte is None else f"<= {dte} DTE"
        print(f"\n=== {snap['symbol']}  {tag}  ({snap['timestamp']}) ===")
        print(f"  spot          {snap['spot']:>14,.2f}")
        print(f"  total GEX     {snap['total_gex']/1e9:>14,.3f} $bn / 1%")
        if snap["gamma_flip"]:
            print(f"  gamma flip    {snap['gamma_flip']:>14,.2f}"
                  f"   (spot {snap['flip_distance_pct']:+.2f}% vs flip)")
        else:
            print("  gamma flip                 none in +/-10% band")
        for name, key in (("call wall", "call_wall"), ("put wall", "put_wall")):
            v = snap[key]
            print(f"  {name:<13s} {v:>14,.0f}" if v is not None
                  else f"  {name:<13s} {'-':>14s}")
        print(f"  contracts     {snap['n_contracts']:>14,}   OI {snap['total_oi']:>12,.0f}")
