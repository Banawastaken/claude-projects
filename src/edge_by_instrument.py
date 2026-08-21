"""Raw edge of the chosen strategy on every instrument, limits switched off.

The challenge simulation answers "did it clear 8% inside the horizon", which
conflates having no edge with having an edge too slow to hit the target. This
measures expectancy per trade instead, so the two can be told apart.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market  # noqa: E402
from evaluate import raw_edge, unlimited_rules  # noqa: E402
from run_instruments import load_instrument, rules_for  # noqa: E402
from universe import UNIVERSE  # noqa: E402

import strategies_final as F  # noqa: E402

WINDOWS = [("DEV", "2025-02-01", "2025-12-01"), ("TEST", "2025-12-01", None)]


def window_idx(df, a, b):
    m = df["ts"] >= pd.Timestamp(a, tz="UTC")
    if b:
        m &= df["ts"] < pd.Timestamp(b, tz="UTC")
    idx = np.flatnonzero(m.values)
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else (0, 0)


if __name__ == "__main__":
    rows = []
    for inst in UNIVERSE:
        df = load_instrument(inst)
        if df is None or len(df) < 3000:
            continue
        base = rules_for(inst, df)
        mkt = Market(df)
        rec = {"instrument": inst.fn_name, "class": inst.asset_class}
        for label, a, b in WINDOWS:
            i0, i1 = window_idx(df, a, b)
            if i1 - i0 < 500:
                continue
            r = unlimited_rules(base)
            e = raw_edge(F.A3_DonchianH4(), mkt, i0, i1, r)
            rec[f"n_{label}"] = e["trades"]
            rec[f"exp_{label}"] = e["expectancy_R"]
            rec[f"totR_{label}"] = e["total_R"]
            rec[f"pf_{label}"] = e["pf"]
        rec["spread_bp"] = mkt.median_spread / float(df["close"].median()) * 10000
        rows.append(rec)
        print(f"  {inst.fn_name:8s} DEV exp {rec.get('exp_DEV', 0):+.3f} "
              f"(n={rec.get('n_DEV', 0):3d}, totR {rec.get('totR_DEV', 0):+6.1f})   "
              f"TEST exp {rec.get('exp_TEST', 0):+.3f} "
              f"(n={rec.get('n_TEST', 0):3d}, totR {rec.get('totR_TEST', 0):+6.1f})   "
              f"spread {rec['spread_bp']:.2f}bp", flush=True)
    out = pd.DataFrame(rows)
    out["exp_avg"] = out[["exp_DEV", "exp_TEST"]].mean(axis=1)
    out["totR_sum"] = out[["totR_DEV", "totR_TEST"]].sum(axis=1)
    out = out.sort_values("exp_avg", ascending=False)
    out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "reports", "instrument_edge.csv"), index=False)
    print("\n=== ranked by average expectancy across both windows ===")
    show = out[["instrument", "class", "n_DEV", "exp_DEV", "n_TEST", "exp_TEST",
                "exp_avg", "totR_sum", "spread_bp"]].copy()
    for c in ("exp_DEV", "exp_TEST", "exp_avg"):
        show[c] = show[c].round(3)
    show["totR_sum"] = show["totR_sum"].round(1)
    show["spread_bp"] = show["spread_bp"].round(2)
    print(show.to_string(index=False))
