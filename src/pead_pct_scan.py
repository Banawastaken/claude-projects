"""Every surprise threshold from 0.00% to 10.00%, and what the winner is worth.

A thousand and one levels over four hundred and eighty-three trades will
produce a best cell, and it will look good. The question this answers is not
which level wins -- something always wins -- but whether winning means
anything, so the scan is run twice: once on the real pairing of surprise to
outcome, and a thousand times on pairings deliberately broken by shuffling the
surprises across trades. The shuffled runs cannot contain an edge, so the best
cell they reach is the score a meaningless parameter earns on this grid. If
the real winner sits inside that distribution, it is the same thing.

Speed comes from the book being separable under flat weighting: each trade's
daily contribution is computed once, and a threshold is then just a choice of
which columns to add. Sorting trades by surprise makes every threshold a
prefix of one ordering, so a single cumulative sum prices the entire scan.
The construction is checked against the ordinary backtest before it is used.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multistrat import periods_per_year, stats  # noqa: E402
from pead_concordant import backtest, build, sides  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")
GRID = np.round(np.arange(0.0, 10.0001, 0.01), 2)   # percent, two decimals
DRAWS = 1000
SEED = 20260829


def trade_matrix(df, rets, sessions, side, hold=60):
    """One column of daily net returns per trade.

    Returns decompose exactly under flat weighting; costs do not quite. When a
    position closes on the same day another opens in the same name, the real
    book nets the two and pays one round of turnover, while separate columns
    pay both. So the decomposition charges marginally more than the book ever
    would -- it errs against the strategy, never for it, which is the only
    direction worth tolerating. Both properties are asserted: gross to the
    tick, net never flattering.
    """
    whole = backtest(df, rets, sessions, side, hold=hold)
    idx = np.flatnonzero(side != 0)
    net, gross = [], []
    for j in idx:
        one = np.zeros(len(side))
        one[j] = side[j]
        bt = backtest(df, rets, sessions, one, hold=hold)
        net.append(bt["ret"].to_numpy())
        gross.append(bt["gross"].to_numpy())
    R = np.column_stack(net)

    g = float(np.abs(np.column_stack(gross).sum(axis=1)
                     - whole["gross"].to_numpy()).max())
    assert g < 1e-12, f"gross does not decompose: {g}"

    slack = R.sum(axis=1) - whole["ret"].to_numpy()
    assert slack.max() < 1e-12, f"decomposition flatters the book by {slack.max()}"
    drag = float(whole["ret"].sum() - R.sum())
    print(f"  cost of decomposing per trade: {drag*1e4:.2f} bp over the sample"
          f"  (same-day swaps the real book would net)")
    return R, df.loc[idx].reset_index(drop=True), whole["ret"].index


def scan(R, pct, ppy):
    """Sharpe at every grid level, via one cumulative sum over sorted trades.

    Trades enter the book as the threshold falls, so ordering them by
    descending surprise makes each level a prefix and the whole scan a single
    pass rather than a thousand backtests.
    """
    order = np.argsort(-pct)
    C = np.cumsum(R[:, order], axis=1)                  # days x prefix length
    kept = np.searchsorted(-pct[order], -GRID / 100.0, side="right")
    out = np.full(len(GRID), np.nan)
    live = kept > 0
    sub = C[:, kept[live] - 1]
    mu, sd = sub.mean(axis=0), sub.std(axis=0)
    out[live] = np.where(sd > 0, mu / sd * np.sqrt(ppy), np.nan)
    return out, kept


def main():
    df, rets, sessions = build(hold=60, market_adjust=True)
    side = sides(df, concordant=True)
    R, tr, idx = trade_matrix(df, rets, sessions, side)
    pct = pd.to_numeric(tr["surprise_pct"], errors="coerce").to_numpy()
    pct = np.abs(np.nan_to_num(pct, nan=-1.0))          # unmeasurable never clears a bar
    ppy = periods_per_year(idx)

    sh, kept = scan(R, pct, ppy)
    best = int(np.nanargmax(sh))

    print("=" * 74)
    print("1. THE SCAN: 0.00% to 10.00%, every hundredth of a percent")
    print("=" * 74)
    print(f"  grid levels tried              : {len(GRID)}")
    print(f"  distinct portfolios they make  : {len(np.unique(kept))}")
    print("    (a level only changes anything when a trade sits between it and")
    print("     the last one, so most of the thousand are duplicates)")
    print(f"\n  best level  : {GRID[best]:.2f}%   Sharpe {sh[best]:.3f}"
          f"   ({kept[best]} of {R.shape[1]} trades)")
    print(f"  no filter   : 0.00%   Sharpe {sh[0]:.3f}   ({kept[0]} trades)")

    print("\n  the neighbourhood of the winner:")
    print(f"  {'level':>8s}{'trades':>8s}{'Sharpe':>9s}")
    for g in range(max(0, best - 5), min(len(GRID), best + 6)):
        mark = "  <-- best" if g == best else ""
        print(f"  {GRID[g]:>7.2f}%{kept[g]:>8d}{sh[g]:>9.3f}{mark}")

    print("\n  how jagged is the surface? a real edge gives a plateau:")
    d = np.abs(np.diff(sh[~np.isnan(sh)]))
    print(f"    mean absolute change between adjacent levels : {d.mean():.4f}")
    print(f"    largest single-level jump                    : {d.max():.4f}")
    print(f"    Sharpe range across the whole grid           : "
          f"{np.nanmin(sh):.3f} to {np.nanmax(sh):.3f}")

    print()
    print("=" * 74)
    print("2. WHAT A MEANINGLESS PARAMETER SCORES ON THE SAME GRID")
    print("=" * 74)
    rng = np.random.default_rng(SEED)
    null = np.empty(DRAWS)
    for i in range(DRAWS):
        null[i] = np.nanmax(scan(R, rng.permutation(pct), ppy)[0])
    pctile = float((null < sh[best]).mean() * 100)
    print(f"  surprises shuffled across trades, {DRAWS} times, best cell kept")
    print(f"    median best Sharpe from noise : {np.median(null):.3f}")
    print(f"    5th to 95th percentile        : "
          f"{np.percentile(null, 5):.3f} to {np.percentile(null, 95):.3f}")
    print(f"    the real winner               : {sh[best]:.3f}"
          f"   -> {pctile:.0f}th percentile of noise")
    print(f"  p = {(1 - pctile / 100):.3f} that shuffled data beats it")

    print()
    print("=" * 74)
    print("3. THE ONLY TEST THAT COUNTS: choose on design, spend on holdout")
    print("=" * 74)
    m = idx < SPLIT
    sh_d, _ = scan(R[m], pct, periods_per_year(idx[m]))
    pick = int(np.nanargmax(sh_d))
    hold_mask = idx >= SPLIT
    sh_h, kept_h = scan(R[hold_mask], pct, periods_per_year(idx[hold_mask]))

    rows = [("best on design", GRID[pick]), ("best on holdout", GRID[int(np.nanargmax(sh_h))]),
            ("no filter", 0.0), ("the 5% you asked about", 5.0)]
    print(f"  {'level':<26s}{'chosen at':>11s}{'design':>9s}{'holdout':>9s}")
    print("  " + "-" * 55)
    for name, lv in rows:
        g = int(np.argmin(np.abs(GRID - lv)))
        print(f"  {name:<26s}{GRID[g]:>10.2f}%{sh_d[g]:>9.2f}{sh_h[g]:>9.2f}")

    r = np.corrcoef(sh_d[~np.isnan(sh_d) & ~np.isnan(sh_h)],
                    sh_h[~np.isnan(sh_d) & ~np.isnan(sh_h)])[0, 1]
    print(f"\n  correlation of the two surfaces across the grid: {r:+.2f}")
    print("  Near +1 would mean a level that worked keeps working.")

    s = stats(pd.Series(R[:, pct >= GRID[pick] / 100.0].sum(axis=1), index=idx))
    print(f"\n  the design-chosen level over the whole sample:"
          f"  CAGR {s['cagr']*100:.2f}%  Sharpe {s['sharpe']:.2f}  t {s['t']:.2f}")


if __name__ == "__main__":
    main()
