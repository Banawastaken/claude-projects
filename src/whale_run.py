"""Run the block-print framework on a stored MBO file."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whale import (add_context, fmt, forward, minute_table,  # noqa: E402
                   rth_mask, summarise, trades_from_mbo)

# ES is $50 a point, NQ $20.
POINT_VALUE = {"ES": 50.0, "NQ": 20.0}


def main(mbo_path, sec_path, symbol="ES", cache="data/mbo/{s}_minutes.parquet"):
    cache = cache.format(s=symbol)
    if os.path.exists(cache):
        bar = pd.read_parquet(cache)
        print(f"cached {len(bar):,} minute bars")
    else:
        # Cache the raw trades too: streaming them is the expensive step, and a
        # bug anywhere downstream should not cost another pass over 2 GiB.
        traw = cache.replace("_minutes", "_trades")
        if os.path.exists(traw):
            tr = pd.read_parquet(traw)
            print(f"cached {len(tr):,} trades")
        else:
            tr = trades_from_mbo(mbo_path)
            tr.to_parquet(traw)
            print(f"{len(tr):,} trades -> {traw}")
        bar = minute_table(tr)
        bar = add_context(bar, sec_path)
        bar.to_parquet(cache)
        print(f"{len(bar):,} minute bars -> {cache}")

    pv = POINT_VALUE[symbol]

    # Is `side` the aggressor? If so, signed print size should move the bar.
    same = bar["print_side"] * (bar["close"] - bar["open"])
    print(f"\naggressor check: mean signed same-bar move "
          f"{same.mean():+.4f} pts ({'aggressor' if same.mean() > 0 else 'RESTING — signs inverted'})")

    r = bar[rth_mask(bar.index)]
    print(f"{len(bar):,} minutes total, {len(r):,} in regular hours "
          f"({len(r)/len(bar)*100:.0f}%)")
    print(f"largest-print size: median {bar['print_size'].median():.0f}, "
          f"p90 {bar['print_size'].quantile(0.9):.0f}, max {bar['print_size'].max():.0f}")

    print(f"\n{'='*70}\n1. HORIZON: how long does the market take to digest it?"
          f"\n   prints > 100 contracts, regular hours\n{'='*70}")
    big = r[r["print_size"] > 100]
    rows = [summarise(forward(big, h, bar), pv, f"  hold {h} min") for h in (1, 5, 10, 15, 20, 30, 60)]
    print(fmt(rows))

    print(f"\n{'='*70}\n2. SIZE: does the edge scale, then break down?"
          f"\n   20-minute hold, regular hours\n{'='*70}")
    bands = [(0, 50, "under 50 (noise)"), (50, 70, "50-70"), (70, 100, "70-100"),
             (100, 190, "100-190 (sweet spot)"), (190, 200, "190-200"),
             (200, 1e9, "over 200 (breakdown)")]
    rows = []
    for lo, hi, name in bands:
        sub = r[(r["print_size"] >= lo) & (r["print_size"] < hi)]
        rows.append(summarise(forward(sub, 20, bar), pv, f"  {name}"))
    print(fmt(rows))

    print(f"\n{'='*70}\n3. FILTERS: session, wick location, book balance"
          f"\n   prints > 100 contracts, 20-minute hold\n{'='*70}")
    rows = [
        summarise(forward(bar[bar["print_size"] > 100], 20, bar), pv, "  all hours"),
        summarise(forward(big, 20, bar), pv, "  regular hours only"),
        summarise(forward(big[big["at_extreme"]], 20, bar), pv, "  + at candle extreme"),
        summarise(forward(big[big["balanced"]], 20, bar), pv, "  + balanced book"),
        summarise(forward(big[big["at_extreme"] & big["balanced"]], 20, bar), pv,
                  "  + both filters"),
    ]
    print(fmt(rows))
    print("\n  net is after crossing one tick; his reported figure is "
          "$53/trade at PF 1.24")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else "data/mbo/GLBX.MDP3_ES.c.0_mbo_2026-08-10_2026-08-17.dbn.zst",
         sys.argv[2] if len(sys.argv) > 2 else "data/mbo/es_1s_v2.parquet",
         sys.argv[3] if len(sys.argv) > 3 else "ES")
