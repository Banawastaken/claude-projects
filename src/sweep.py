"""Parameter sweeps on the development window.

The point is to find broad plateaus, not peaks. A setting that only works at one
exact value is curve-fit; a setting that works across a neighbourhood is an
edge. Every sweep here runs on 2025 data only, and the winners are checked once
against the untouched Dec-2025-onward test window.
"""

from __future__ import annotations

import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market  # noqa: E402
from evaluate import raw_edge  # noqa: E402
from run import load, slice_period  # noqa: E402

_MKT = {}


def _get(tag, start, end):
    key = (tag, start, end)
    if key not in _MKT:
        df = load(tag)
        _MKT[key] = (Market(df), *slice_period(df, start, end))
    return _MKT[key]


def variant(base_cls, **params):
    """Build a subclass of base_cls with attributes overridden."""
    name = base_cls.name + " " + " ".join(f"{k}={v}" for k, v in params.items())
    return type(base_cls.__name__ + "_v", (base_cls,), {**params, "name": name})


def _job(args):
    mod, cls_name, params, tag, start, end = args
    import importlib

    m = importlib.import_module(mod)
    cls = variant(getattr(m, cls_name), **params)
    mkt, i0, i1 = _get(tag, start, end)
    e = raw_edge(cls(), mkt, i0, i1)
    return {**params, **e}


def sweep(mod, cls_name, grid: dict, tag="xauusd_m1_clean",
          start="2025-02-01", end="2025-12-01", workers=4) -> pd.DataFrame:
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    jobs = [(mod, cls_name, c, tag, start, end) for c in combos]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_job, jobs):
            rows.append(r)
    df = pd.DataFrame(rows)
    return df.sort_values("expectancy_R", ascending=False)


def show(df: pd.DataFrame, n=25, keys=None):
    cols = (keys or []) + ["trades", "win_rate", "expectancy_R", "total_R", "pf", "max_dd_pct"]
    cols = [c for c in cols if c in df.columns]
    d = df[cols].head(n).copy()
    d["win_rate"] = (d["win_rate"] * 100).round(1)
    d["expectancy_R"] = d["expectancy_R"].round(3)
    d["total_R"] = d["total_R"].round(1)
    d["pf"] = d["pf"].round(2)
    d["max_dd_pct"] = d["max_dd_pct"].round(1)
    print(d.to_string(index=False))
