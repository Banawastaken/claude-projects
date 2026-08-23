"""Dump the backtest into a compact JSON the report page can render.

Curves are sampled to month ends: eleven years of daily points is far more
resolution than a 700px chart can show, and the file has to be embedded in the
page rather than fetched.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from multistrat import (benchmarks, combine, contribution,  # noqa: E402
                        drawdown_table, stats)
from yearly import yearly_returns  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")


def curve(r: pd.Series):
    """Month-end equity curve, indexed to 100 at the start."""
    cum = (1 + r).cumprod() * 100.0
    m = cum.resample("ME").last().dropna()
    return [[d.strftime("%Y-%m"), round(float(v), 2)] for d, v in m.items()]


def _clean(d):
    return {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v))
                else round(float(v), 4)) for k, v in d.items()}


def pead_block():
    """PEAD's standalone result plus the drift-by-liquidity table behind it."""
    try:
        from pead import build_events, causal_percentile, run as pead_run
    except Exception:
        return None
    bands = (("micro", 0, 5e6), ("liquid", 5e6, None))
    out = {"traded": {}, "drift": []}
    for name, lo, hi in bands:
        g = pead_run(min_adv=lo, max_adv=hi, apply_costs=False)
        n = pead_run(min_adv=lo, max_adv=hi, apply_costs=True)
        yrs = (n["ret"].index[-1] - n["ret"].index[0]).days / 365.25
        out["traded"][name] = {
            "gross": _clean(stats(g["ret"])), "net": _clean(stats(n["ret"])),
            "trades": int(len(n["trades"])),
            "names": int(n["trades"]["ticker"].nunique()),
            "fees_pa": round(float(n["fees"].sum() / yrs) * 100, 2)}

    df0, excess, _ = build_events()
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
        return v.mean(), v.std(ddof=1) / np.sqrt(len(v))

    for label, lo, hi in (("ADV < $5M", 0, 5e6), ("$5M-$100M", 5e6, 1e8),
                          ("ADV > $100M", 1e8, np.inf)):
        d = df0[(df0["adv"] >= lo) & (df0["adv"] < hi)].reset_index(drop=True)
        if len(d) < 500:
            continue
        d["pct"] = causal_percentile(d)
        d = d.dropna(subset=["pct"]).reset_index(drop=True)
        row = {"band": label, "events": int(len(d)),
               "names": int(d["ticker"].nunique())}
        for h in (20, 60, 120):
            mt, st = drift(d[d["pct"] >= 0.8], h)
            mb, sb = drift(d[d["pct"] <= 0.2], h)
            sp = mt - mb
            se = float(np.sqrt(st ** 2 + sb ** 2))
            row[f"d{h}"] = round(sp * 100, 2)
            row[f"t{h}"] = round(sp / se, 2) if se > 0 else None
        out["drift"].append(row)
    return out


def filter_block():
    """PEAD's traded result under each filter combination, micro-cap band."""
    try:
        from pead import run as pead_run
    except Exception:
        return None
    out = []
    for label, im, vc in (("Unfiltered", None, None),
                          ("Least volatile 25%", None, 0.25),
                          ("Buyback-aligned", "aligned", None),
                          ("Both filters", "aligned", 0.25)):
        try:
            g = pead_run(min_adv=0, max_adv=5e6, apply_costs=False,
                         issuance_mode=im, vol_cut=vc)
            n = pead_run(min_adv=0, max_adv=5e6, apply_costs=True,
                         issuance_mode=im, vol_cut=vc)
        except Exception:
            continue
        out.append({"label": label, "gross": _clean(stats(g["ret"])),
                    "net": _clean(stats(n["ret"]))})
    return out


def orderflow_block(path="data/mbo/es_1s_v2.parquet"):
    """Order-flow features on real ES data, and the economics of trading them."""
    import os
    if not os.path.exists(path):
        return None
    sys.path.insert(0, "src")
    from mbo_features import predictive_test
    from mbo_run import rth

    f = pd.read_parquet(path)
    f = rth(f[f["updates"] > 0])
    pt = predictive_test(f, horizons=(1, 5, 30, 60))
    mid = f["mid"].to_numpy()
    sp = f["spread"].to_numpy()
    qi = f["qi"].to_numpy()

    econ = []
    for h in (1, 5, 30):
        fwd = np.full(len(mid), np.nan)
        fwd[:-h] = mid[h:] / mid[:-h] - 1.0
        for th in (0.3, 0.6, 0.9):
            sig = np.where(qi > th, 1.0, np.where(qi < -th, -1.0, 0.0))
            m = (sig != 0) & np.isfinite(fwd)
            if m.sum() < 200:
                continue
            gross = sig[m] * fwd[m]
            cost = sp[m] / mid[m]
            net = gross - cost
            econ.append({"h": h, "threshold": th, "trades": int(m.sum()),
                         "gross_bp": round(float(gross.mean()) * 1e4, 4),
                         "cost_bp": round(float(cost.mean()) * 1e4, 4),
                         "net_bp": round(float(net.mean()) * 1e4, 4)})

    return {
        "bars": int(len(f)),
        "quotes": 24323530,
        "spend_usd": 6.96,
        "mean_spread_bp": round(float((sp / mid).mean()) * 1e4, 3),
        "sec_vol_bp": round(float(pd.Series(mid).pct_change().std()) * 1e4, 3),
        "predictive": [{k: (round(float(v), 4) if isinstance(v, float) else v)
                        for k, v in r.items()} for r in pt.to_dict("records")],
        "economics": econ,
    }


def main(src="data/multistrat/sleeves.parquet", out="data/multistrat/viz.json"):
    f = pd.read_parquet(src)
    r_iv, w_iv = combine(f, "invvol")
    # Equal weight is reported alongside, not swapped in: inverse volatility
    # was the allocation rule chosen up front, and picking the better-scoring
    # of the two after seeing both is selection on the outcome.
    r_eq, _ = combine(f, "equal")

    payload = {
        "window": [str(f.index.min().date()), str(f.index.max().date())],
        "sleeves": list(f.columns),
        "curves": {c: curve(f[c]) for c in f.columns},
        "combined": curve(r_iv),
        "stats": {c: {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4))
                      for k, v in stats(f[c]).items()} for c in f.columns},
        "stats_active": {c: {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4))
                             for k, v in stats(f[c][f[c].abs() > 1e-12]).items()}
                         for c in f.columns},
        "stats_combined": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4))
                           for k, v in stats(r_iv).items()},
        "stats_design": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4))
                         for k, v in stats(r_iv[r_iv.index < SPLIT]).items()},
        "stats_holdout": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4))
                          for k, v in stats(r_iv[r_iv.index >= SPLIT]).items()},
        "benchmarks": {k: {kk: (None if vv is None or (isinstance(vv, float) and not np.isfinite(vv)) else round(float(vv), 4))
                           for kk, vv in v.items()} for k, v in benchmarks().items()},
        "stats_equal": {k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else round(float(v), 4))
                        for k, v in stats(r_eq).items()},
        "pead": pead_block(),
        "filters": filter_block(),
        "orderflow": orderflow_block(),
        "yearly": {},
        "corr": {a: {b: round(float(f[a].corr(f[b])), 3) for b in f.columns}
                 for a in f.columns},
        "contribution": {k: round(float(v), 4) for k, v in contribution(f, w_iv).items()},
        "drawdowns": [[str(s.date()), str(e.date()),
                       (str(b.date()) if b is not None else None), round(d, 4)]
                      for s, e, b, d in drawdown_table(r_iv)],
    }

    y = yearly_returns(f) * 100
    y["COMBINED"] = (r_iv.groupby(r_iv.index.year)
                     .apply(lambda g: (1 + g).prod() - 1) * 100)
    payload["yearly"] = {str(int(k)): {c: round(float(v), 2)
                                       for c, v in row.items()}
                         for k, row in y.iterrows()}

    with open(out, "w") as fh:
        json.dump(payload, fh)
    print(f"wrote {out}  ({len(json.dumps(payload)):,} bytes)")
    return payload


if __name__ == "__main__":
    main()
