"""Leave-one-out sensitivity of the concordant PEAD result.

Eleven names is few enough that one of them can carry the headline. This drops
each in turn and re-runs, which is the cheapest honest check on whether a
Sharpe ratio is a property of the strategy or of the sample.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "src")

from multistrat import stats  # noqa: E402
from pead_concordant import backtest, build, sides  # noqa: E402


def main(market_adjust=True):
    df, rets, sess = build(market_adjust=market_adjust)
    names = sorted(df["ticker"].unique())
    base = stats(backtest(df, rets, sess, sides(df, True))["ret"])
    print(f"all {len(names)} names: Sharpe {base['sharpe']:.2f}, "
          f"t {base['t']:.2f}, CAGR {base['cagr']*100:.2f}%\n")

    rows = []
    for n in names:
        d = df[df["ticker"] != n].reset_index(drop=True)
        s = stats(backtest(d, rets, sess, sides(d, True))["ret"])
        rows.append((n, s["sharpe"], s["t"], s["cagr"]))

    print(f"  {'dropped':<8s}{'Sharpe':>9s}{'t':>8s}{'CAGR':>9s}{'shift':>9s}")
    print("  " + "-" * 42)
    for n, sh, t, c in sorted(rows, key=lambda x: x[1]):
        print(f"  {n:<8s}{sh:>9.2f}{t:>8.2f}{c*100:>8.2f}%{sh-base['sharpe']:>+9.2f}")
    arr = np.array([r[1] for r in rows])
    print(f"\n  Sharpe spans {arr.min():.2f} to {arr.max():.2f} on the removal of "
          f"a single name.")
    print("  A number that moves that far on one stock is a property of the "
          "sample,\n  not yet of the strategy.")
    return rows


if __name__ == "__main__":
    main()
