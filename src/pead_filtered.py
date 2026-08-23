"""PEAD with the issuance and volatility filters, measured across the grid.

Every combination is reported. The point is not to find the cell that works but
to see whether any cell works on data it was not chosen on, and to keep the
number of cells visible next to the answer.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pead import DEFAULT_MIN_ADV, build_events, causal_percentile  # noqa: E402
from pead_filters import attach_issuance, attach_volatility, load_shares  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")

ISSUANCE = {
    "none": None,
    "buyback long / dilute short": "aligned",
    "exclude diluters": "exclude",
}
VOLBANDS = {"all": None, "least volatile 25%": 0.25, "least volatile 50%": 0.50}


def drift_fn(excess):
    ex = excess.fillna(0.0)
    cols = {t: j for j, t in enumerate(ex.columns)}
    A = ex.to_numpy()

    def drift(sub, h):
        vals = []
        for r in sub.itertuples():
            j = cols.get(r.ticker)
            if j is None:
                continue
            a, b = r.entry_i, min(r.entry_i + h, len(A))
            if b > a:
                vals.append(np.prod(1 + A[a:b, j]) - 1)
        v = np.array(vals)
        if len(v) < 30:
            return np.nan, np.nan, len(v)
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v)), len(v)

    return drift


def legs(df, issuance_mode, vol_cut, top=0.2):
    """Which events go long and which go short under one filter combination."""
    d = df
    if vol_cut is not None:
        d = d[d["vol_rank"] <= vol_cut]
    if len(d) < 200:
        return None, None
    d = d.copy()
    d["pct"] = causal_percentile(d)
    d = d.dropna(subset=["pct"])

    long_ = d[d["pct"] >= 1 - top]
    short = d[d["pct"] <= top]
    if issuance_mode == "aligned":
        # Two independent signals pointing the same way: a good reaction from a
        # company shrinking its share count, against a bad one from a diluter.
        long_ = long_[long_["issuance"] < 0]
        short = short[short["issuance"] > 0]
    elif issuance_mode == "exclude":
        long_ = long_[long_["issuance"] < long_["issuance"].median()]
        short = short[short["issuance"] > short["issuance"].median()]
    return long_, short


def main(horizon=120, min_adv=DEFAULT_MIN_ADV, max_adv=None, band="liquid"):
    df, excess, sessions = build_events()
    shares = load_shares()
    print(f"share histories for {len(shares)} names")
    df = df[df["adv"] >= min_adv]
    if max_adv is not None:
        df = df[df["adv"] < max_adv]
    df = df.reset_index(drop=True)
    df = attach_issuance(df, shares)
    df = attach_volatility(df, excess, sessions)
    have = df["issuance"].notna()
    print(f"{len(df):,} {band} events, {int(have.sum()):,} with issuance known "
          f"({have.mean()*100:.0f}%)\n")
    df = df[have].reset_index(drop=True)

    dates = pd.DatetimeIndex(sessions)[df["react_i"].to_numpy()]
    df["react_date"] = dates
    drift = drift_fn(excess)

    hdr = (f"{'issuance filter':<28s}{'volatility':<22s}"
           f"{'n L/S':>13s}{'design':>10s}{'t':>7s}{'holdout':>10s}{'t':>7s}")
    print(f"D+{horizon} top-minus-bottom spread, design 2015-2020 vs holdout 2021-2026")
    print(hdr)
    print("-" * len(hdr))

    cells = 0
    survivors = []
    for iname, imode in ISSUANCE.items():
        for vname, vcut in VOLBANDS.items():
            L, S = legs(df, imode, vcut)
            if L is None or len(L) < 60 or len(S) < 60:
                print(f"{iname:<28s}{vname:<22s}{'too few':>13s}")
                continue
            cells += 1
            row = f"{iname:<28s}{vname:<22s}{len(L):>6d}/{len(S):<6d}"
            vals = {}
            for tag, lo, hi in (("design", pd.Timestamp.min, SPLIT),
                                ("holdout", SPLIT, pd.Timestamp.max)):
                Lw = L[(L["react_date"] >= lo) & (L["react_date"] < hi)]
                Sw = S[(S["react_date"] >= lo) & (S["react_date"] < hi)]
                mt, st, nt = drift(Lw, horizon)
                mb, sb, nb = drift(Sw, horizon)
                if not np.isfinite(mt) or not np.isfinite(mb):
                    vals[tag] = (np.nan, np.nan)
                    continue
                sp = mt - mb
                se = float(np.sqrt(st ** 2 + sb ** 2))
                vals[tag] = (sp, sp / se if se > 0 else np.nan)
            for tag in ("design", "holdout"):
                sp, t = vals.get(tag, (np.nan, np.nan))
                row += (f"{sp*100:>9.2f}%{t:>7.2f}" if np.isfinite(sp)
                        else f"{'-':>10s}{'-':>7s}")
            print(row)
            d_sp, d_t = vals["design"]
            h_sp, h_t = vals["holdout"]
            # Both windows must carry the effect. Requiring only the holdout
            # lets through cells whose design window was flat or negative,
            # which is a sign flip dressed as a discovery.
            if (np.isfinite(d_sp) and np.isfinite(h_sp)
                    and d_t > 1.0 and h_t > 1.0):
                survivors.append((iname, vname, d_sp, h_sp, h_t))

    print("-" * len(hdr))
    print(f"{cells} filter combinations measured. At the 5% level about "
          f"{cells*0.05:.1f} would clear\nt=2 on one window by chance, and about "
          f"{cells*0.05*0.05:.2f} would clear it on both.")
    if survivors:
        print("\ncells positive on both windows with holdout t > 1:")
        for s in survivors:
            print(f"  {s[0]} / {s[1]}: design {s[2]*100:+.2f}%, "
                  f"holdout {s[3]*100:+.2f}% (t={s[4]:.2f})")
    else:
        print("\nno cell is positive on both windows with holdout t > 1.")


if __name__ == "__main__":
    h = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    band = sys.argv[2] if len(sys.argv) > 2 else "liquid"
    if band == "micro":
        main(h, min_adv=0.0, max_adv=DEFAULT_MIN_ADV, band="micro-cap")
    else:
        main(h, band="liquid")
