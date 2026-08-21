"""Parameter sweep for the v6 concepts, design window only.

Scores on the whole design window per instrument first (cheap), then re-checks
the leaders year by year, because a good aggregate with a bad year profile is
the failure mode this whole exercise exists to avoid.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market  # noqa: E402
from evaluate import raw_edge, unlimited_rules  # noqa: E402
from run_instruments import rules_for  # noqa: E402
from screen_breadth import DESIGN_YEARS, load, year_slice  # noqa: E402
from universe import BY_FN  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def variant(base, **params):
    return type(base.__name__ + "_v", (base,), {**params, "name": base.name})


def design_slice(df):
    m = (df["ts"] >= pd.Timestamp("2015-01-01", tz="UTC")) & \
        (df["ts"] < pd.Timestamp("2021-01-01", tz="UTC"))
    idx = np.flatnonzero(m.values)
    return int(idx[0]), int(idx[-1]) + 1


def sweep(base, grid, names):
    insts = [BY_FN[n] for n in names]
    ctx = []
    for inst in insts:
        df = load(inst)
        if df is None:
            continue
        ctx.append((inst, df, Market(df), unlimited_rules(rules_for(inst, df))))

    keys = list(grid)
    rows = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, vals))
        cls = variant(base, **params)
        tot_r, trades, per_inst = 0.0, 0, []
        for inst, df, mkt, rules in ctx:
            i0, i1 = design_slice(df)
            e = raw_edge(cls(), mkt, i0, i1, rules)
            tot_r += e["total_R"]
            trades += e["trades"]
            per_inst.append(e["total_R"])
        rows.append({**params, "trades": trades,
                     "expR": tot_r / trades if trades else 0.0,
                     "totR": tot_r,
                     "inst_pos": sum(1 for v in per_inst if v > 0),
                     "n_inst": len(per_inst)})
    return pd.DataFrame(rows).sort_values("expR", ascending=False)


def year_profile(base, params, names):
    """Per-year expectancy for one configuration, across the design years."""
    cls = variant(base, **params)
    out = {}
    for n in names:
        inst = BY_FN[n]
        df = load(inst)
        if df is None:
            continue
        mkt = Market(df)
        rules = unlimited_rules(rules_for(inst, df))
        for y in DESIGN_YEARS:
            i0, i1 = year_slice(df, y)
            if i1 - i0 < 400:
                continue
            e = raw_edge(cls(), mkt, i0, i1, rules)
            out.setdefault(y, []).append(e["expectancy_R"])
    return {y: float(np.mean(v)) for y, v in sorted(out.items())}


if __name__ == "__main__":
    import strategies_v6 as V6

    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["XAUUSD", "EURUSD", "GBPUSD"]
    print(f"sweeping ShortTermReversal on {names}, design window 2015-2020\n")
    grid = {
        "pull_bars": [2, 3, 4],
        "tp_r": [1.5, 2.5, 4.0],
        "atr_mult": [1.5, 2.5],
        "trend_len": [50, 100, 200],
    }
    df = sweep(V6.ShortTermReversal, grid, names)
    cols = list(grid) + ["trades", "expR", "totR", "inst_pos"]
    print(df[cols].head(18).round(3).to_string(index=False))

    best = df.iloc[0]
    params = {k: best[k] for k in grid}
    params = {k: (int(v) if k in ("pull_bars", "trend_len") else float(v))
              for k, v in params.items()}
    print(f"\nyear profile for the leader {params}:")
    prof = year_profile(V6.ShortTermReversal, params, names)
    print("  " + "  ".join(f"{y}:{v:+.3f}" for y, v in prof.items()))
    print(f"  positive years: {sum(1 for v in prof.values() if v > 0)}/{len(prof)}")
