"""The block-print framework on six months of NQ, from trades + BBO."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from whale import (fmt, forward, minute_table, qi_from_bbo,  # noqa: E402
                   rth_mask, summarise, trades_from_trades_file)

TRADES = "data/mbo/GLBX.MDP3_NQ.c.0_trades_2026-02-20_2026-08-20.dbn.zst"
BBO = "data/mbo/GLBX.MDP3_NQ.c.0_bbo-1s_2026-02-20_2026-08-20.dbn.zst"
CACHE = "data/mbo/NQ_minutes.parquet"
POINT_VALUE = 20.0     # NQ is $20 a point


def build():
    if os.path.exists(CACHE):
        bar = pd.read_parquet(CACHE)
        print(f"cached {len(bar):,} minute bars")
        return bar
    traw = "data/mbo/NQ_trades.parquet"
    if os.path.exists(traw):
        tr = pd.read_parquet(traw)
        print(f"cached {len(tr):,} trades")
    else:
        tr = trades_from_trades_file(TRADES)
        tr.to_parquet(traw)
        print(f"{len(tr):,} trades")
    bar = minute_table(tr)

    rng = (bar["high"] - bar["low"]).replace(0, np.nan)
    pos = (bar["print_price"] - bar["low"]) / rng
    bar["wick_pos"] = pos
    bar["at_extreme"] = np.where(bar["print_side"] > 0, pos >= 0.8,
                                 np.where(bar["print_side"] < 0, pos <= 0.2, False))

    if os.path.exists(BBO):
        qi = qi_from_bbo(BBO)
        bar["qi"] = qi.resample("1min").last().reindex(bar.index, method="ffill")
    else:
        bar["qi"] = np.nan
    bar["balanced"] = bar["qi"].abs() <= 0.3
    bar.to_parquet(CACHE)
    print(f"{len(bar):,} minute bars -> {CACHE}")
    return bar


def main():
    bar = build()
    pv = POINT_VALUE
    same = bar["print_side"] * (bar["close"] - bar["open"])
    print(f"\naggressor check: {same.mean():+.4f} pts "
          f"({'aggressor' if same.mean() > 0 else 'RESTING — inverted'})")
    r = bar[rth_mask(bar.index)]
    big_all = bar[bar["print_size"] > 100]
    big = r[r["print_size"] > 100]
    print(f"{len(bar):,} minutes, {len(r):,} RTH. "
          f"prints>100: {len(big_all):,} total, {len(big):,} RTH "
          f"({len(big)/max(len(big_all),1)*100:.0f}% — he says 79%)")
    print(f"largest-print size: median {bar['print_size'].median():.0f}, "
          f"p90 {bar['print_size'].quantile(0.9):.0f}, "
          f"p99 {bar['print_size'].quantile(0.99):.0f}, max {bar['print_size'].max():.0f}")

    print(f"\n{'='*74}\n1. HORIZON  (prints > 100, RTH)\n{'='*74}")
    print(fmt([summarise(forward(big, h, bar), pv, f"  hold {h} min")
               for h in (1, 5, 10, 15, 20, 30, 60)]))

    print(f"\n{'='*74}\n2. SIZE  (20-minute hold, RTH)\n{'='*74}")
    bands = [(0, 50, "under 50 (noise)"), (50, 70, "50-70"), (70, 100, "70-100"),
             (100, 190, "100-190 (sweet spot)"), (190, 200, "190-200"),
             (200, 1e9, "over 200 (breakdown)")]
    print(fmt([summarise(forward(r[(r["print_size"] >= lo) & (r["print_size"] < hi)], 20, bar),
                         pv, f"  {name}") for lo, hi, name in bands]))

    print(f"\n{'='*74}\n3. FILTERS  (prints > 100, 20-minute hold)\n{'='*74}")
    print(fmt([
        summarise(forward(big_all, 20, bar), pv, "  all hours"),
        summarise(forward(big, 20, bar), pv, "  regular hours only"),
        summarise(forward(big[big["at_extreme"]], 20, bar), pv, "  + at candle extreme"),
        summarise(forward(big[big["balanced"]], 20, bar), pv, "  + balanced book"),
        summarise(forward(big[big["at_extreme"] & big["balanced"]], 20, bar), pv,
                  "  + both (his full stack)"),
    ]))
    print("\n  his reported result: $53 per trade, profit factor 1.24")

    print(f"\n{'='*74}\n4. HIS FULL STACK, split in half by date\n{'='*74}")
    full = big[big["at_extreme"] & big["balanced"]]
    if len(full) > 60:
        mid = full.index[len(full) // 2]
        print(fmt([
            summarise(forward(full[full.index < mid], 20, bar), pv, "  first half"),
            summarise(forward(full[full.index >= mid], 20, bar), pv, "  second half"),
        ]))
    else:
        print(f"  only {len(full)} trades survive the full stack — too few to split")


if __name__ == "__main__":
    main()
