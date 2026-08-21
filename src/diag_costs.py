"""Is there an edge that costs destroy, or no edge to begin with?

These are very different conclusions. If the gross signal earns +0.10 R and
lands at zero after spread and commission, the problem is the cost base and a
bigger account or a cheaper broker changes the answer. If the gross signal is
also flat, the rules simply do not predict anything and no amount of cost
reduction helps.

Frictionless mode sets ask equal to bid and zeroes commission and slippage.
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
from screen_breadth import load  # noqa: E402
from universe import BY_FN  # noqa: E402

NAMES = ["XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "AUDUSD",
         "USDCAD", "USDJPY", "SPX500", "NDX100"]


def frictionless(df):
    """Same bars with the spread collapsed to zero."""
    d = df.copy()
    for side in ("open", "high", "low", "close"):
        d[f"ask_{side}"] = d[side]
    d["spread_med"] = 0.0
    return d


def design_slice(df):
    m = (df["ts"] >= pd.Timestamp("2015-01-01", tz="UTC")) & \
        (df["ts"] < pd.Timestamp("2021-01-01", tz="UTC"))
    idx = np.flatnonzero(m.values)
    return int(idx[0]), int(idx[-1]) + 1


if __name__ == "__main__":
    import strategies_final as F
    import strategies_v6 as V6

    concepts = [(V6.ShortTermReversal, {"pull_bars": 2, "tp_r": 2.5,
                                        "atr_mult": 2.5, "trend_len": 200}),
                (V6.TSMomentum, {}), (V6.TurnOfMonth, {}),
                (V6.VolContraction, {}), (F.A3_DonchianH4, {})]

    print("Gross (no costs) versus net expectancy, design window, 9 instruments\n")
    print(f"{'concept':22s} {'trades':>7s} {'gross expR':>11s} {'net expR':>9s} "
          f"{'cost/R':>7s}")
    for cls, params in concepts:
        c = type(cls.__name__ + "_v", (cls,), {**params, "name": cls.name}) if params else cls
        g_tot = n_tot = trades = 0.0
        for name in NAMES:
            inst = BY_FN[name]
            df = load(inst)
            if df is None:
                continue
            i0, i1 = design_slice(df)

            mkt_net = Market(df)
            r_net = unlimited_rules(rules_for(inst, df))
            e_net = raw_edge(c(), mkt_net, i0, i1, r_net)

            mkt_g = Market(frictionless(df))
            r_g = unlimited_rules(rules_for(inst, df))
            r_g.commission_per_lot = 0.0
            r_g.slip_entry_spread = 0.0
            r_g.slip_stop_spread = 0.0
            e_g = raw_edge(c(), mkt_g, i0, i1, r_g)

            g_tot += e_g["total_R"]
            n_tot += e_net["total_R"]
            trades += e_net["trades"]
        if trades == 0:
            continue
        gross, net = g_tot / trades, n_tot / trades
        print(f"{cls.name[:22]:22s} {int(trades):7d} {gross:+11.3f} {net:+9.3f} "
              f"{gross - net:7.3f}", flush=True)
