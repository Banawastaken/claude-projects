"""Screen concepts by breadth across instruments and years.

The previous strategy had a superb average and no breadth: it made everything
in two years on one market. So the metric here is the fraction of
instrument-year cells with positive expectancy, not the mean. A concept that
earns +0.05 R almost everywhere is worth far more than one averaging +0.5 R in
two cells out of a hundred.

Design window only (2015-2020). Held-out years are not read.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market  # noqa: E402
from evaluate import raw_edge, unlimited_rules  # noqa: E402
from run_instruments import rules_for  # noqa: E402
from universe import UNIVERSE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DECADE = os.path.join(HERE, "..", "data", "decade")
REPORTS = os.path.join(HERE, "..", "reports")

DESIGN_YEARS = list(range(2015, 2021))


def load(inst):
    p = os.path.join(DECADE, f"{inst.fn_name}.parquet")
    if not os.path.exists(p):
        return None
    return pd.read_parquet(p).sort_values("ts").reset_index(drop=True)


def year_slice(df, year):
    m = (df["ts"] >= pd.Timestamp(f"{year}-01-01", tz="UTC")) & \
        (df["ts"] < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
    idx = np.flatnonzero(m.values)
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) > 400 else (0, 0)


def screen(concepts, instruments=None, years=None):
    years = years or DESIGN_YEARS
    insts = instruments or UNIVERSE
    cells = []
    for inst in insts:
        df = load(inst)
        if df is None or len(df) < 5000:
            continue
        rules = unlimited_rules(rules_for(inst, df))
        mkt = Market(df)
        for cls in concepts:
            for y in years:
                i0, i1 = year_slice(df, y)
                if i1 - i0 < 400:
                    continue
                e = raw_edge(cls(), mkt, i0, i1, rules)
                if e["trades"] < 5:
                    continue
                cells.append({
                    "concept": cls.name, "instrument": inst.fn_name,
                    "class": inst.asset_class, "year": y,
                    "trades": e["trades"], "expR": e["expectancy_R"],
                    "totR": e["total_R"], "pf": e["pf"],
                })
        print(f"  screened {inst.fn_name}", flush=True)
    return pd.DataFrame(cells)


def summarise(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, g in cells.groupby("concept"):
        by_inst = g.groupby("instrument")["totR"].sum()
        rows.append({
            "concept": name,
            "cells": len(g),
            "breadth%": (g["expR"] > 0).mean() * 100,
            "instruments+%": (by_inst > 0).mean() * 100,
            "trades/yr": g["trades"].mean(),
            "expR_med": g["expR"].median(),
            "expR_mean": g["expR"].mean(),
            "totR": g["totR"].sum(),
        })
    return pd.DataFrame(rows).sort_values("breadth%", ascending=False)


if __name__ == "__main__":
    import strategies_final as F
    import strategies_v6 as V6

    concepts = list(V6.CANDIDATES) + [F.A3_DonchianH4]   # A3 as the control
    print(f"Screening {len(concepts)} concepts on {len(DESIGN_YEARS)} design years\n")
    cells = screen(concepts)
    if cells.empty:
        print("no data yet")
        sys.exit(1)
    os.makedirs(REPORTS, exist_ok=True)
    cells.to_csv(os.path.join(REPORTS, "screen_cells.csv"), index=False)

    s = summarise(cells)
    print("\n=== breadth on the design window (2015-2020) ===")
    print("breadth% = share of instrument-year cells with positive expectancy\n")
    print(s.round(2).to_string(index=False))

    print("\n=== by concept and asset class (breadth %) ===")
    piv = cells.assign(pos=cells["expR"] > 0).pivot_table(
        index="concept", columns="class", values="pos", aggfunc="mean") * 100
    print(piv.round(0).to_string())
