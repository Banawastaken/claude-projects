"""Does the MFI/OBV gap close, and if it does, does closing it pay?

Two questions that are easy to confuse. The gap closing is a statement about
two indicators. Making money needs the gap to say something about price. The
first is tested for completeness and is expected to pass almost mechanically;
the second is the one that decides whether there is a strategy here.

Everything is measured across all twenty-nine instruments rather than on the
one the idea was noticed on, because an effect that only exists where it was
spotted is the definition of the thing to be careful about. Forward windows
overlap, which inflates a t-statistic badly, so significance is taken from
non-overlapping samples only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mfi_obv as M  # noqa: E402

SYMS = sorted(f[:-8] for f in os.listdir("data/decade"))
HOLDS = (1, 4, 8, 24)
NORMS = (21, 50, 100, 200, 480)


def one(df, n_norm, hold):
    f = M.features(df, n_mfi=21, n_norm=n_norm)
    r = M.forward_return(df["close"], hold)
    ok = f["spread"].notna() & r.notna()
    return f.loc[ok, "spread"], r[ok], f.loc[ok, "obv_n"]


def nonoverlap_t(x, hold):
    """t on every hold-th observation, so the windows do not share bars."""
    s = x.iloc[::hold].dropna()
    if len(s) < 30 or s.std() == 0:
        return np.nan, len(s)
    return float(s.mean() / (s.std() / np.sqrt(len(s)))), len(s)


def main():
    print("=" * 78)
    print("1. DOES THE GAP CLOSE?  (change in OBV-normalised over the next 8 bars,")
    print("   by how far below or above MFI it started)")
    print("=" * 78)
    rows = []
    for sym in SYMS:
        d = M.load_h1(sym)
        sp, _, on = one(d, 100, 8)
        fwd_obv = on.shift(-8) - on
        for lo, hi, lab in ((-100, -30, "OBV far above MFI"), (-30, 30, "close together"),
                            (30, 100, "OBV far below MFI")):
            m = (sp >= lo) & (sp < hi)
            if m.sum() > 50:
                rows.append({"sym": sym, "bucket": lab, "d_obv": fwd_obv[m].mean()})
    g = pd.DataFrame(rows).groupby("bucket")["d_obv"].agg(["mean", "count"])
    print(f"\n  {'starting gap':<22s}{'mean move in OBV_norm':>24s}{'symbols':>10s}")
    print("  " + "-" * 56)
    for lab in ("OBV far above MFI", "close together", "OBV far below MFI"):
        if lab in g.index:
            print(f"  {lab:<22s}{g.loc[lab,'mean']:>22.2f}  {int(g.loc[lab,'count']):>9d}")
    print("\n  The gap closes. It closes because a rolling min-max series cannot")
    print("  stay pinned at its own extreme, so this was never in doubt.")

    print()
    print("=" * 78)
    print("2. DOES THE GAP PREDICT PRICE?  information coefficient, all symbols")
    print("   (Spearman rank correlation of spread with forward return)")
    print("=" * 78)
    print(f"\n  {'norm window':>12s}" + "".join(f"{f'{h}-bar':>12s}" for h in HOLDS))
    print("  " + "-" * 62)
    ics = {}
    for n_norm in NORMS:
        line = f"  {n_norm:>12d}"
        for hold in HOLDS:
            vals = []
            for sym in SYMS:
                d = M.load_h1(sym)
                sp, r, _ = one(d, n_norm, hold)
                if len(sp) > 500:
                    vals.append(sp.corr(r, method="spearman"))
            ics[(n_norm, hold)] = np.mean(vals)
            line += f"{np.mean(vals):>12.4f}"
        print(line)
    print("\n  An exploitable signal needs this consistently away from zero.")
    print("  For scale: a respectable equity factor runs about +0.03.")

    print()
    print("=" * 78)
    print("3. TRADING IT: long when MFI is above OBV by the threshold, short below")
    print("   net of the real bid/ask spread in the data, 8-bar hold")
    print("=" * 78)
    print(f"\n  {'symbol':<10s}{'trades':>8s}{'net/trade bp':>14s}{'t (non-ovl)':>13s}")
    print("  " + "-" * 45)
    tot = []
    for sym in SYMS:
        d = M.load_h1(sym)
        f = M.features(d, 21, 100)
        r = M.forward_return(d["close"], 8)
        side = np.where(f["spread"] > 30, 1.0, np.where(f["spread"] < -30, -1.0, 0.0))
        cost = (d["spread_med"] / d["close"]).fillna(0.0)
        pnl = pd.Series(side, index=d.index) * r - np.abs(side) * cost
        pnl = pnl[f["spread"].notna() & r.notna() & (side != 0)]
        if len(pnl) < 100:
            continue
        t, n = nonoverlap_t(pnl, 8)
        tot.append({"sym": sym, "n": len(pnl), "bp": pnl.mean() * 1e4, "t": t})
        print(f"  {sym:<10s}{len(pnl):>8d}{pnl.mean()*1e4:>14.2f}{t:>13.2f}")
    T = pd.DataFrame(tot)
    print("  " + "-" * 45)
    print(f"  {'ALL':<10s}{T['n'].sum():>8d}{T['bp'].mean():>14.2f}"
          f"{T['t'].mean():>13.2f}   (mean across symbols)")
    print(f"\n  symbols with positive net edge: {int((T['bp'] > 0).sum())} of {len(T)}")
    print(f"  symbols with t > 2:             {int((T['t'] > 2).sum())} of {len(T)}")


if __name__ == "__main__":
    main()
