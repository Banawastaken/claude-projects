"""Daily dealer gamma levels for NDX, rebuilt from the OPRA chains.

One row per session: total gamma exposure, the gamma flip, and the call and put
walls -- the levels both channels describe as where price is expected to turn.

Everything is computed from data known before the session it is keyed to.
Open interest is stamped pre-open and describes the previous close; the
settlement prices implied volatility comes from are the previous close's too.
So the levels for day D are what a trader could actually have drawn on the
chart at D's open, which is the only version worth testing.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gex as G  # noqa: E402
from opra_chain import definitions, implied_vol_vec, open_interest, settles  # noqa: E402

OUT = "data/opra"
NY = "America/New_York"


def spot_series(path="data/decade/NDX100.parquet"):
    """Daily NDX level, from the H1 CFD bars already in the project."""
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[(df["high"] > df["low"]) | (df["volume"] > 0)]
    g = df.groupby(df["ts"].dt.normalize())
    return pd.DataFrame({
        "close": 0.5 * (g["close"].last() + g["ask_close"].last()),
        "high": g["high"].max(), "low": g["low"].min(),
        "open": 0.5 * (g["open"].first() + g["ask_open"].first()),
    })


def fit_surface(k, sqt, iv):
    """Least-squares implied vol surface: quadratic in moneyness, linear in root-time.

    Only about 390 of the 2,300 NDX contracts carrying open interest trade on
    any given day, so a settlement price exists for 36% of the open interest
    and the missing 64% sits away from the money -- precisely where the walls
    are. Pricing gamma from the traded subset alone measures a third of the
    book and calls it the whole.

    Five parameters fitted on ~390 points is well determined, and the shape is
    the standard one: a smile in log-moneyness that flattens with maturity.
    """
    X = np.column_stack([np.ones_like(k), k, k ** 2, sqt, k * sqt])
    beta, *_ = np.linalg.lstsq(X, iv, rcond=None)
    resid = iv - X @ beta
    ss = float(1 - resid.var() / iv.var()) if iv.var() > 0 else np.nan
    return beta, ss


def apply_surface(beta, k, sqt):
    X = np.column_stack([np.ones_like(k), k, k ** 2, sqt, k * sqt])
    return np.clip(X @ beta, 0.03, 2.5)


def build(out=os.path.join(OUT, "ndx_gamma_levels.parquet"), min_oi=1,
          max_dte=None):
    if os.path.exists(out):
        return pd.read_parquet(out)

    defs = definitions().set_index("instrument_id")
    oi = open_interest()
    px = settles()
    spot = spot_series()

    # Settlement prices are the previous close's, so they pair with the open
    # interest of the same stamp: both describe the state entering the day.
    ch = oi.merge(px, on=["date", "instrument_id"], how="inner")
    ch = ch.join(defs, on="instrument_id", how="inner")
    ch = ch[ch["oi"] >= min_oi]
    ch["dte"] = (ch["expiry"] - ch["date"]).dt.days
    ch = ch[(ch["dte"] >= 0) & (ch["dte"] <= 400)]

    sp = spot["close"].copy()
    sp.index = pd.DatetimeIndex(sp.index)
    ch["spot"] = sp.reindex(ch["date"]).to_numpy()
    ch = ch.dropna(subset=["spot"])
    ch = ch[(ch["close"] > 0.05) & (ch["strike"] > 0)]

    ch["iv"] = implied_vol_vec(ch["close"], ch["spot"], ch["strike"],
                               ch["dte"] / 365.0, ch["right"])
    ch = ch.dropna(subset=["iv"])
    ch = ch[(ch["iv"] > 0.01) & (ch["iv"] < 3.0)]
    print(f"{len(ch):,} contract-days with a usable implied vol across "
          f"{ch['date'].nunique()} sessions", flush=True)

    # Every contract carrying open interest, priced off the fitted surface.
    full = oi.join(defs, on="instrument_id", how="inner")
    full = full[full["oi"] >= min_oi]
    full["dte"] = (full["expiry"] - full["date"]).dt.days
    full = full[(full["dte"] >= 0) & (full["dte"] <= 400)]
    full["spot"] = sp.reindex(full["date"]).to_numpy()
    full = full.dropna(subset=["spot"])
    full = full[full["strike"] > 0]

    # One global surface per day, plus a dedicated smile for each expiry that
    # has enough traded strikes. The smile shape changes a lot with maturity --
    # steep and kinked at the front, flat at the back -- so a single surface
    # across all expiries fits the near-dated wings badly, which is where the
    # walls sit.
    fits, smiles = {}, {}
    for d, g in ch.groupby("date"):
        if len(g) < 25:
            continue
        k = np.log(g["strike"].to_numpy(float) / g["spot"].to_numpy(float))
        sqt = np.sqrt(np.maximum(g["dte"].to_numpy(float), 1.0) / 365.0)
        fits[d] = fit_surface(k, sqt, g["iv"].to_numpy(float))
        for e, ge in g.groupby("expiry"):
            if len(ge) < 8:
                continue
            ke = np.log(ge["strike"].to_numpy(float) / ge["spot"].to_numpy(float))
            X = np.column_stack([np.ones_like(ke), ke, ke ** 2])
            beta, *_ = np.linalg.lstsq(X, ge["iv"].to_numpy(float), rcond=None)
            smiles[(d, e)] = beta
    r2 = np.array([v[1] for v in fits.values()])
    print(f"vol surface fitted on {len(fits)} sessions (median R^2 "
          f"{np.nanmedian(r2):.3f}), plus {len(smiles):,} per-expiry smiles",
          flush=True)

    rows = []
    for d, g in full.groupby("date"):
        if d not in fits:
            continue
        beta, _ = fits[d]
        s = float(g["spot"].iloc[0])
        kk = np.log(g["strike"].to_numpy(float) / s)
        sqt = np.sqrt(np.maximum(g["dte"].to_numpy(float), 1.0) / 365.0)
        iv = apply_surface(beta, kk, sqt)
        # Prefer the expiry's own smile wherever one was fitted.
        exp_arr = g["expiry"].to_numpy()
        for e in np.unique(exp_arr):
            b = smiles.get((d, pd.Timestamp(e)))
            if b is None:
                continue
            m = exp_arr == e
            ke = kk[m]
            iv[m] = np.clip(b[0] + b[1] * ke + b[2] * ke ** 2, 0.03, 2.5)
        arrs = {"strike": g["strike"].to_numpy(float),
                "dte": g["dte"].to_numpy(float),
                "right": g["right"].to_numpy(float),
                "oi": g["oi"].to_numpy(float),
                "iv": iv}
        grid, tot = G.profile(arrs, s, max_dte=max_dte)
        flip = G.gamma_flip(grid, tot)
        ks, agg = G.by_strike(arrs, s, max_dte=max_dte)
        cw, pw = G.walls(ks, agg, s)
        rows.append({
            "date": d, "spot": s,
            "total_gex": float(G.gex_at(s, arrs, max_dte=max_dte)[0].sum()),
            "gamma_flip": flip, "call_wall": cw, "put_wall": pw,
            "contracts": int(len(g)), "total_oi": float(g["oi"].sum()),
        })
    lv = pd.DataFrame(rows).set_index("date").sort_index()
    lv.to_parquet(out)
    print(f"wrote {out}: {len(lv)} sessions", flush=True)
    return lv


if __name__ == "__main__":
    lv = build()
    print(lv.describe().to_string())
    print()
    print(lv.tail(5).to_string())
