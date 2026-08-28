"""Rank the basket name by name, then ask whether ranking is worth anything.

Picking the names with the best Sharpe and combining only those is guaranteed
to produce a good backtest. The number it produces is not evidence, because the
same returns chose the names and then scored them. The only question that
matters is whether a name that looked good in the past goes on looking good,
so everything here is built around one test: rank on the design window alone,
then spend the holdout on the names that ranking picked.

Three baselines make the answer readable -- keeping every name, keeping the
names the ranking rejected, and drawing the same number of names at random a
few thousand times. If selection carries information, the picked subset should
beat the random draw more often than not. If it does not, ranking is an
expensive way to sample noise.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multistrat import stats  # noqa: E402
from pead_concordant import backtest, build, sides  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")
SEED = 20260828


def per_name(hold=60, market_adjust=True):
    """One net return series per ticker, on the shared session index.

    Flat weighting makes the book separable: every position contributes its own
    weight times its own return, so the basket's return is the sum of the parts
    and any subset can be priced by adding up the members. That is asserted
    below rather than assumed.
    """
    df, rets, sessions = build(hold=hold, market_adjust=market_adjust)
    side = sides(df, concordant=True)
    whole = backtest(df, rets, sessions, side, hold=hold)

    out = {}
    for tk in sorted(df["ticker"].unique()):
        m = (df["ticker"] == tk).to_numpy()
        sub = backtest(df[m].reset_index(drop=True), rets, sessions,
                       side[m], hold=hold)
        if sub is not None:
            out[tk] = sub["ret"]

    combined = pd.DataFrame(out).sum(axis=1)
    gap = float((combined - whole["ret"]).abs().max())
    assert gap < 1e-12, f"per-name returns do not add up to the basket: {gap}"
    return pd.DataFrame(out), df, side


def window(r, lo=None, hi=None):
    if lo is not None:
        r = r[r.index >= lo]
    if hi is not None:
        r = r[r.index < hi]
    return r


def table(parts, trades, lo=None, hi=None):
    rows = []
    for tk in parts.columns:
        s = stats(window(parts[tk], lo, hi))
        if s.get("n", 0) < 20:
            continue
        rows.append({"ticker": tk, "sharpe": s["sharpe"], "t": s["t"],
                     "cagr": s["cagr"], "trades": int((trades == tk).sum())})
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def main():
    parts, df, side = per_name()
    traded = df.loc[side != 0, "ticker"]

    full = table(parts, traded)
    print("=" * 72)
    print("1. EVERY NAME ON ITS OWN, whole sample, market-adjusted")
    print("=" * 72)
    print(f"  {'ticker':<8s}{'trades':>7s}{'CAGR':>9s}{'Sharpe':>9s}{'t':>8s}")
    print("  " + "-" * 39)
    for r in full.itertuples():
        print(f"  {r.ticker:<8s}{r.trades:>7d}{r.cagr*100:>8.2f}%"
              f"{r.sharpe:>9.2f}{r.t:>8.2f}")

    print()
    print("=" * 72)
    print("2. PICK THE BEST ON THE WHOLE SAMPLE, then score on the same sample")
    print("   (this is the number the idea promises, and it is circular)")
    print("=" * 72)
    for k in (3, 5, 8, 12):
        picked = list(full.head(k)["ticker"])
        s = stats(parts[picked].sum(axis=1))
        print(f"  top {k:>2d} by Sharpe   CAGR {s['cagr']*100:>6.2f}%   "
              f"Sharpe {s['sharpe']:>5.2f}   t {s['t']:>5.2f}   {', '.join(picked)}")
    s = stats(parts.sum(axis=1))
    print(f"  all {len(parts.columns):>2d}            CAGR {s['cagr']*100:>6.2f}%   "
          f"Sharpe {s['sharpe']:>5.2f}   t {s['t']:>5.2f}")

    print()
    print("=" * 72)
    print("3. THE HONEST VERSION: rank on 2015-2020, spend 2021-2026")
    print("=" * 72)
    design = table(parts, traded, hi=SPLIT)
    rng = np.random.default_rng(SEED)
    names = list(parts.columns)

    print(f"  {'':<16s}{'CAGR':>9s}{'Sharpe':>9s}{'t':>8s}{'vs random':>12s}")
    print("  " + "-" * 54)
    for k in (3, 5, 8, 12):
        picked = list(design.head(k)["ticker"])
        worst = list(design.tail(k)["ticker"])
        hold_r = window(parts[picked].sum(axis=1), lo=SPLIT)
        s = stats(hold_r)

        # Where does the picked subset sit against subsets of the same size
        # drawn at random? Selection is only worth something above chance.
        draws = np.array([
            stats(window(parts[list(rng.choice(names, k, replace=False))]
                         .sum(axis=1), lo=SPLIT))["sharpe"]
            for _ in range(3000)])
        pct = float((draws < s["sharpe"]).mean() * 100)

        sw = stats(window(parts[worst].sum(axis=1), lo=SPLIT))
        print(f"  best {k:>2d} on design{s['cagr']*100:>8.2f}%{s['sharpe']:>9.2f}"
              f"{s['t']:>8.2f}{pct:>10.0f}th")
        print(f"  worst {k:>2d} on design{sw['cagr']*100:>7.2f}%{sw['sharpe']:>9.2f}"
              f"{sw['t']:>8.2f}")
    s = stats(window(parts.sum(axis=1), lo=SPLIT))
    print(f"  {'all ' + str(len(names)):<16s}{s['cagr']*100:>8.2f}%"
          f"{s['sharpe']:>9.2f}{s['t']:>8.2f}")

    print()
    print("=" * 72)
    print("4. DOES A NAME'S RANK PERSIST AT ALL?")
    print("=" * 72)
    hold_tab = table(parts, traded, lo=SPLIT).set_index("ticker")
    d = design.set_index("ticker")
    both = d.join(hold_tab, lsuffix="_design", rsuffix="_holdout").dropna()
    rho = both["sharpe_design"].corr(both["sharpe_holdout"], method="spearman")
    print(f"  rank correlation of per-name Sharpe, design vs holdout: {rho:+.2f}"
          f"  (n={len(both)})")
    print("  +1 would mean the ranking is fully informative, 0 that it is noise.")


if __name__ == "__main__":
    main()
