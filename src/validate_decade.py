"""Does A3's edge exist outside the twenty months it was built in?

Everything so far -- development and test -- sits inside Jan 2025 to Aug 2026,
which was an exceptional two-way trending market for gold. This runs the
strategy, untouched, across every year from 2015 and reports the edge and the
challenge outcomes per year.

The data is hourly, and hourly bars flatter this strategy: on the overlapping
period the same code reports +0.620 R per trade on H1 against +0.264 on minute
data, because the trail ratchets once an hour rather than once a minute. Read
every number here as an optimistic bound. A year that loses on H1 would lose
harder in reality.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market, Rules  # noqa: E402
from evaluate import pass_rate_parallel, raw_edge, unlimited_rules  # noqa: E402
from run_instruments import rules_for  # noqa: E402
from universe import BY_FN  # noqa: E402

import strategies_final as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "instruments")
TAG = "INSTRUMENT:XAUUSD_H1_DECADE"


def year_bounds(df, year):
    m = (df["ts"] >= pd.Timestamp(f"{year}-01-01", tz="UTC")) & \
        (df["ts"] < pd.Timestamp(f"{year + 1}-01-01", tz="UTC"))
    idx = np.flatnonzero(m.values)
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) > 500 else (0, 0)


def starts_in(df, first, last, every_days=10):
    ts = df["ts"].values.astype("datetime64[m]")
    t0 = np.datetime64(pd.Timestamp(first, tz="UTC").tz_localize(None), "m")
    t1 = np.datetime64(pd.Timestamp(last, tz="UTC").tz_localize(None), "m")
    out, cur = [], t0
    while cur < t1:
        i = int(np.searchsorted(ts, cur))
        if i < len(ts):
            out.append(i)
        cur = cur + np.timedelta64(every_days, "D")
    return out


if __name__ == "__main__":
    path = os.path.join(DATA, "XAUUSD_H1_DECADE.parquet")
    df = pd.read_parquet(path).sort_values("ts").reset_index(drop=True)
    inst = BY_FN["XAUUSD"]
    base = rules_for(inst, df)
    mkt = Market(df)
    print(f"{len(df):,} H1 bars, {df['ts'].min().date()} -> {df['ts'].max().date()}\n")

    # ---- raw edge per year ------------------------------------------------
    print("Raw edge per year (limits off)\n")
    print(f"{'year':>5s} {'trades':>7s} {'WR%':>6s} {'expR':>7s} {'totR':>8s} "
          f"{'PF':>6s} {'maxDD%':>7s}")
    rows = []
    for y in range(2015, 2027):
        i0, i1 = year_bounds(df, y)
        if i1 - i0 < 500:
            continue
        e = raw_edge(F.A3_DonchianH4(), mkt, i0, i1, unlimited_rules(base))
        rows.append({"year": y, **e})
        print(f"{y:5d} {e['trades']:7d} {e['win_rate'] * 100:6.1f} "
              f"{e['expectancy_R']:7.3f} {e['total_R']:8.1f} {e['pf']:6.2f} "
              f"{e['max_dd_pct']:7.1f}", flush=True)

    ed = pd.DataFrame(rows)
    if not ed.empty:
        pos = (ed["expectancy_R"] > 0).sum()
        print(f"\nprofitable years: {pos}/{len(ed)}   "
              f"mean expR {ed['expectancy_R'].mean():+.3f}   "
              f"median {ed['expectancy_R'].median():+.3f}")
        pre = ed[ed["year"] <= 2024]
        if not pre.empty:
            print(f"2015-2024 only:   {(pre['expectancy_R'] > 0).sum()}/{len(pre)} "
                  f"positive, mean expR {pre['expectancy_R'].mean():+.3f}, "
                  f"total {pre['total_R'].sum():+.1f} R")
        ed.to_csv(os.path.join(HERE, "..", "reports", "decade_edge.csv"), index=False)

    # ---- challenge outcomes, rolling starts across the decade -------------
    print("\n\nChallenge outcomes by era (205-day horizon, starts every 10 days)\n")
    print(f"{'era':12s} {'runs':>5s} {'P1%':>5s} {'fund%':>6s} {'alive%':>7s} "
          f"{'breach%':>8s} {'ddMed':>6s} {'wdMax':>6s} {'paid$':>7s}")
    eras = [("2015-2017", "2015-02-01", "2017-06-01"),
            ("2018-2020", "2018-01-01", "2020-06-01"),
            ("2021-2022", "2021-01-01", "2022-06-01"),
            ("2023-2024", "2023-01-01", "2024-06-01"),
            ("2025-2026", "2025-01-01", "2026-02-01")]
    out_rows = []
    for label, a, b in eras:
        st = starts_in(df, a, b, every_days=10)
        if len(st) < 5:
            continue
        r = pass_rate_parallel("strategies_final", "A3_DonchianH4", {}, st, 205,
                               tag=TAG,
                               rule_over={"contract_size": base.contract_size,
                                          "commission_per_lot": base.commission_per_lot,
                                          "min_lot": base.min_lot})
        if not r:
            continue
        out_rows.append({"era": label, **{k: v for k, v in r.items()
                                          if k not in ("breaches", "detail")}})
        print(f"{label:12s} {r['runs']:5d} {r['p1_pass'] * 100:5.0f} "
              f"{r['funded'] * 100:6.0f} {r['funded_alive'] * 100:7.0f} "
              f"{r['breach_rate'] * 100:8.0f} {r['dd_med']:6.2f} "
              f"{r['worstday_max']:6.2f} {r['payout_when_funded']:7.0f}", flush=True)
    if out_rows:
        pd.DataFrame(out_rows).to_csv(
            os.path.join(HERE, "..", "reports", "decade_challenges.csv"), index=False)
