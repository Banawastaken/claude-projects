"""Build every replicable sleeve, run them together, and report.

Windows follow RESEARCH_PROTOCOL.md: 2015-2020 is the window the rules were
chosen on, 2021-2026 is held back and every number is re-computed on it
unchanged.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from multistrat import (align, benchmarks, combine, contribution,  # noqa: E402
                        drawdown_table, fmt_stats, inverse_vol_weights, stats)
from yearly import yearly_table  # noqa: E402
import sleeves as S  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")


def tsmom_universe():
    """Whatever the decade download has finished, so the run is never blocked."""
    have = sorted(os.path.splitext(f)[0]
                  for f in os.listdir("data/decade") if f.endswith(".parquet"))
    return have


def build():
    out, detail = {}, {}
    print("building sleeves")

    r, frame = S.sleeve_overnight()
    out["overnight_index"] = r
    detail["overnight_index"] = frame
    print(f"  overnight_index    {len(r):>5,} obs  ({', '.join(frame.columns)})")

    r, _ = S.sleeve_gold_january()
    out["gold_january"] = r
    print(f"  gold_january       {len(r):>5,} obs")

    r, frame = S.sleeve_faber()
    if r is not None:
        out["faber_taa"] = r
        detail["faber_taa"] = frame
        print(f"  faber_taa          {len(r):>5,} obs  ({', '.join(frame.columns)})")

    uni = tsmom_universe()
    r, frame = S.sleeve_tsmom(uni)
    if r is not None:
        out["ts_momentum"] = r
        detail["ts_momentum"] = frame
        print(f"  ts_momentum        {len(r):>5,} obs  "
              f"({frame.shape[1]} markets: {', '.join(frame.columns)})")
    return out, detail


def report_pead():
    """PEAD is measured but deliberately left out of the combination.

    It is negative after costs in both liquidity bands, so adding it as a
    sleeve would subtract from the portfolio. Reporting it separately keeps a
    real, faithfully tested result visible without letting a losing strategy
    into the allocation on the grounds that it was interesting.
    """
    try:
        from pead import run as pead_run
    except Exception:
        return
    print(f"\n{'='*104}\nPEAD, measured but not allocated\n{'='*104}")
    rows = []
    for label, lo, hi in (("micro <$5M ADV", 0, 5e6), ("liquid >$5M ADV", 5e6, None)):
        try:
            gross = pead_run(min_adv=lo, max_adv=hi, apply_costs=False)
            net = pead_run(min_adv=lo, max_adv=hi, apply_costs=True)
        except Exception as e:
            print(f"  {label}: unavailable ({e})")
            continue
        yrs = (net["ret"].index[-1] - net["ret"].index[0]).days / 365.25
        rows.append((f"{label}  gross", stats(gross["ret"])))
        rows.append((f"{label}  after cost", stats(net["ret"])))
        print(f"  {label}: {len(net['trades']):,} trades, "
              f"fees {net['fees'].sum()/yrs*100:.2f}%/yr")
    if rows:
        print(fmt_stats(rows))
        print("  Left out of the combination: negative after costs in both bands.")


def report(frame, label):
    print(f"\n{'='*104}\n{label}\n{'='*104}")
    rows = [(c, stats(frame[c][frame[c].abs() > 1e-12])) for c in frame.columns]
    print("per sleeve, measured only on the days it was actually in the market:")
    print(fmt_stats(rows))

    print("\nsame sleeves as an account line (flat days count as zero):")
    print(fmt_stats([(c, stats(frame[c])) for c in frame.columns]))

    r_eq, w_eq = combine(frame, "equal")
    r_iv, w_iv = combine(frame, "invvol")
    r_10, _ = combine(frame, "invvol", target_vol=0.10)
    print("\ncombined:")
    print(fmt_stats([("equal weight", stats(r_eq)),
                     ("inverse vol", stats(r_iv)),
                     ("inverse vol @ 10% vol", stats(r_10))]))
    print("  (the 10% line is the same strategy levered up; leverage moves"
          " return and\n   drawdown together and leaves Sharpe alone.)")

    print("\ncorrelation between sleeves (daily):")
    c = frame.corr()
    print("  " + "".join(f"{x[:11]:>13s}" for x in c.columns))
    for i, row in c.iterrows():
        print(f"  {i[:14]:<14s}" + "".join(f"{v:>13.2f}" for v in row.values))

    contrib = contribution(frame, w_iv)
    print("\nshare of combined return, inverse-vol weighting:")
    for k, v in contrib.items():
        bar = "#" * int(abs(v) * 40)
        print(f"  {k:<20s}{v*100:>7.1f}%  {bar}")
    return r_iv


def main():
    sleeves, detail = build()
    frame = align(sleeves)
    frame = frame[frame.index >= pd.Timestamp("2015-01-01")]

    full = report(frame, "FULL SAMPLE 2015-2026")
    des = report(frame[frame.index < SPLIT], "DESIGN WINDOW 2015-2020")
    hold = report(frame[frame.index >= SPLIT], "HOLDOUT WINDOW 2021-2026")

    print(f"\n{'='*104}\nWORST DRAWDOWNS, combined inverse-vol, full sample\n{'='*104}")
    for start, end, back, depth in drawdown_table(full):
        rec = back.date() if back is not None else "not recovered"
        print(f"  {depth*100:>7.2f}%   {start.date()} -> {end.date()}   "
              f"recovered {rec}")

    report_pead()

    print(f"\n{'='*104}\nPASSIVE BENCHMARKS, same window\n{'='*104}")
    print(fmt_stats(list(benchmarks().items()) +
                    [("this portfolio (inv vol)", stats(full))]))

    print(f"\n{'='*104}\nRETURN BY CALENDAR YEAR\n{'='*104}")
    print("2015 is a warm-up: the inverse-vol allocation needs 60 active days"
          " per sleeve\nbefore it will fund anything, so the combined line"
          " starts flat.\n")
    print(yearly_table(frame, full))

    os.makedirs("data/multistrat", exist_ok=True)
    frame.to_parquet("data/multistrat/sleeves.parquet")
    print("\nwrote data/multistrat/sleeves.parquet")


if __name__ == "__main__":
    main()
