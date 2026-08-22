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
