"""Where the momentum sleeve's return actually comes from.

Breadth is the whole justification for time-series momentum, so it matters
whether the sleeve earns a little from many markets or nearly all of it from
one. Crypto is the specific worry here: BTC rose roughly a hundredfold over
the sample, and a universe defined by what a broker lists today includes it
with hindsight.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from multistrat import stats  # noqa: E402
from sleeves import sleeve_tsmom  # noqa: E402


def main():
    import os
    names = sorted(os.path.splitext(f)[0] for f in os.listdir("data/decade")
                   if f.endswith(".parquet"))
    r, frame = sleeve_tsmom(names)
    tot = frame.sum(axis=0)
    share = tot / tot.abs().sum()

    print(f"time-series momentum across {frame.shape[1]} markets\n")
    print(f"{'market':<12s}{'total %':>10s}{'share':>9s}{'Sharpe':>9s}{'t':>8s}")
    print("-" * 48)
    for n in tot.sort_values(ascending=False).index:
        s = stats(frame[n][frame[n].abs() > 1e-12])
        print(f"{n:<12s}{tot[n]*100:>9.1f}%{share[n]*100:>8.1f}%"
              f"{s.get('sharpe', float('nan')):>9.2f}{s.get('t', float('nan')):>8.2f}")

    print("-" * 48)
    pos = int((tot > 0).sum())
    print(f"{pos}/{len(tot)} markets positive")

    for drop in (["BTCUSD", "ETHUSD"], ["BTCUSD"]):
        keep = [c for c in frame.columns if c not in drop]
        r2 = frame[keep].sum(axis=1)
        s = stats(r2)
        print(f"\nexcluding {', '.join(drop)}: CAGR {s['cagr']*100:.2f}%  "
              f"Sharpe {s['sharpe']:.2f}  t {s['t']:.2f}")
    s = stats(r)
    print(f"full universe:            CAGR {s['cagr']*100:.2f}%  "
          f"Sharpe {s['sharpe']:.2f}  t {s['t']:.2f}")


if __name__ == "__main__":
    main()
