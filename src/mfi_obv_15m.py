"""The MFI/OBV gap on the full 15-minute XAUUSD history, which is the timeframe
the idea was drawn from.

An earlier pass used the 13.5 months that had finished downloading and found a
signal worth having a look at: information coefficient +0.045, gross edge
rising with holding period to about four basis points. That window was
January 2021 to February 2022, a rangebound stretch for gold, and a
mean-reversion idea tested only on a range is being asked a question it cannot
fail. On the full five and a half years the coefficient falls to +0.016 and
the gross edge goes to roughly zero, which is what the shorter window was
hiding.

Costs are charged on both legs of non-overlapping trades. Results are split
into three two-year blocks rather than reported as one number, because a
strategy that works in one regime and not the others should not be allowed to
average itself into looking steady.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mfi_obv as M  # noqa: E402

M1 = "data/xauusd_m1.parquet"
BLOCKS = ((2021, 2022), (2023, 2024), (2025, 2026))


def load_15m(path=M1):
    m1 = pd.read_parquet(path)
    m1 = m1.set_index(pd.DatetimeIndex(m1["ts"]).tz_localize(None)).sort_index()
    m1["spread_med"] = m1["ask_close"] - m1["close"]
    cols = ["open", "high", "low", "close", "volume", "spread_med"]
    return M.resample(m1[cols], "15min")


def main():
    d = load_15m()
    cost = (d["spread_med"] / d["close"]).fillna(0.0)
    print(f"XAUUSD 15M: {len(d):,} bars  {d.index[0].date()} -> {d.index[-1].date()}")
    print(f"median spread {(cost).median()*1e4:.2f} bp per side\n")

    hdr = (f"{'norm':>5s}{'thr':>5s}{'hold':>6s}{'n':>7s}{'gross':>9s}{'cost':>8s}"
           f"{'net':>8s}{'t':>7s}" + "".join(f"{f'{a}-{b%100:02d}':>9s}" for a, b in BLOCKS))
    print(hdr)
    print("-" * len(hdr))
    for n_norm in (21, 100):
        sp = M.features(d, 21, n_norm)["spread"]
        for hold in (24, 48, 96):
            t = M.trades(d, sp, 30, hold, cost)
            if len(t) < 50:
                continue
            tt = t["net"].mean() / (t["net"].std() / np.sqrt(len(t)))
            yr = pd.DatetimeIndex(t["ts"]).year
            segs = [t[(yr >= a) & (yr <= b)]["net"].mean() * 1e4 for a, b in BLOCKS]
            print(f"{n_norm:>5d}{30:>5d}{hold:>6d}{len(t):>7d}"
                  f"{t['gross'].mean()*1e4:>9.2f}{t['cost'].mean()*1e4:>8.2f}"
                  f"{t['net'].mean()*1e4:>8.2f}{tt:>7.2f}"
                  + "".join(f"{s:>9.2f}" for s in segs))

    sp = M.features(d, 21, 100)["spread"]
    r = M.forward_return(d["close"], 8)
    m = sp.notna() & r.notna()
    print(f"\ninformation coefficient, 8-bar: {sp[m].corr(r[m], method='spearman'):+.4f}")


if __name__ == "__main__":
    main()
