"""His PEAD rules, and what each component of them is worth."""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from multistrat import fmt_stats, stats  # noqa: E402
from pead_concordant import backtest, build, sides  # noqa: E402

SPLIT = pd.Timestamp("2021-01-01")


def main(hold=60):
    for adj in (True, False):
        tag = "market-adjusted" if adj else "raw returns"
        df, rets, sessions = build(hold=hold, market_adjust=adj)
        if df is None or df.empty:
            print("no events -- fetch consensus data first")
            return
        if adj:
            print(f"{len(df):,} announcements across {df['ticker'].nunique()} names, "
                  f"{df['date'].min()} .. {df['date'].max()}")
            beat = (df["surprise"] > 0).mean()
            agree = ((df["surprise"] > 0) == (df["reaction"] > 0)).mean()
            print(f"  {beat*100:.0f}% beat consensus; surprise and reaction agree "
                  f"{agree*100:.0f}% of the time")
            print(f"  {df['before_open'].mean()*100:.0f}% released pre-market\n")

        rows = []
        for label, conc in (("concordant (his rule)", True),
                            ("price reaction only", False)):
            s = sides(df, concordant=conc)
            res = backtest(df, rets, sessions, s, hold=hold)
            if res is None:
                continue
            n_tr = len(res["trades"])
            rows.append((f"{label}  net", stats(res["ret"])))
            rows.append((f"{label}  gross", stats(res["gross"])))
            if adj:
                print(f"  {label}: {n_tr:,} trades "
                      f"({int((s > 0).sum())} long, {int((s < 0).sum())} short), "
                      f"median gross exposure {res['exposure'].median()*100:.0f}%")
        print(f"\n=== {tag} ===")
        print(fmt_stats(rows))
        print()

    # design vs holdout on his rule, market-adjusted
    df, rets, sessions = build(hold=hold, market_adjust=True)
    res = backtest(df, rets, sessions, sides(df, True), hold=hold)
    r = res["ret"]
    print("=== his rule, market-adjusted, out of sample ===")
    print(fmt_stats([("design 2015-2020", stats(r[r.index < SPLIT])),
                     ("holdout 2021-2026", stats(r[r.index >= SPLIT]))]))


if __name__ == "__main__":
    main()
