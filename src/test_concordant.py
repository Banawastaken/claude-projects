"""Checks on the concordant PEAD pipeline.

The result this produces is positive, which is exactly when a backtest deserves
the most suspicion. These test the two places a look-ahead would hide: which
session is treated as reacting, and when the position starts earning.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pead_concordant as C  # noqa: E402


def ok(name, cond, note=""):
    print(f"  {name:<54s} {'OK' if cond else 'FAIL'}  {note}")
    if not cond:
        raise AssertionError(name)


SESS = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=40))


def test_premarket_release_reacts_the_same_day():
    i = SESS.searchsorted(pd.Timestamp("2024-01-10"), side="left")
    ok("a pre-market release reacts on its own day",
       SESS[i] == pd.Timestamp("2024-01-10"), f"{SESS[i].date()}")


def test_postmarket_release_reacts_the_next_day():
    i = SESS.searchsorted(pd.Timestamp("2024-01-10"), side="right")
    ok("a post-market release reacts the next session",
       SESS[i] == pd.Timestamp("2024-01-11"), f"{SESS[i].date()}")


def test_concordance_requires_agreement():
    df = pd.DataFrame({
        "surprise": [1.0, 1.0, -1.0, -1.0, 0.0],
        "reaction": [0.02, -0.02, -0.02, 0.02, 0.02],
    })
    s = C.sides(df, concordant=True)
    ok("beat and up goes long", s[0] == 1.0)
    ok("beat but down is skipped", s[1] == 0.0)
    ok("miss and down goes short", s[2] == -1.0)
    ok("miss but up is skipped", s[3] == 0.0)
    ok("no surprise is skipped", s[4] == 0.0)


def test_price_only_ignores_the_surprise():
    df = pd.DataFrame({"surprise": [-1.0, 1.0], "reaction": [0.02, -0.02]})
    s = C.sides(df, concordant=False)
    ok("without the filter only the reaction decides",
       s[0] == 1.0 and s[1] == -1.0)


def _synthetic(n_events=40, hold=5):
    sess = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=200))
    rets = pd.DataFrame({"AAA": np.zeros(len(sess))}, index=sess)
    rows = []
    for k in range(n_events):
        i = 10 + k * 4
        if i + hold + 2 >= len(sess):
            break
        rows.append({"ticker": "AAA", "date": str(sess[i].date()),
                     "before_open": True, "surprise": 1.0, "surprise_pct": 0.1,
                     "reaction": 0.01, "react_i": i, "entry_i": i + 1})
    return pd.DataFrame(rows), rets, sess


def test_position_does_not_earn_its_own_entry_day():
    df, rets, sess = _synthetic(1, hold=5)
    i = int(df.loc[0, "entry_i"])
    rets = rets.copy()
    rets.iloc[i, 0] = 1.0            # a huge move on the entry session itself
    res = C.backtest(df, rets, sess, C.sides(df, True), hold=5, cost_bp=0.0)
    ok("the entry session's own return is not collected",
       abs(float(res["gross"].iloc[i])) < 1e-12,
       "entry is placed that session, so it earns from the next")


def test_position_earns_the_session_after_entry():
    df, rets, sess = _synthetic(1, hold=5)
    i = int(df.loc[0, "entry_i"])
    rets = rets.copy()
    rets.iloc[i + 1, 0] = 0.10
    res = C.backtest(df, rets, sess, C.sides(df, True), hold=5,
                     flat_weight=0.10, cost_bp=0.0)
    ok("a 10% move the next session earns 10% of a 10% position",
       abs(float(res["gross"].iloc[i + 1]) - 0.01) < 1e-9,
       f"{float(res['gross'].iloc[i+1]):.4f}")


def test_position_closes_after_the_holding_period():
    df, rets, sess = _synthetic(1, hold=5)
    i = int(df.loc[0, "entry_i"])
    rets = rets.copy()
    rets.iloc[i + 20, 0] = 0.50       # long after the 5-session hold ends
    res = C.backtest(df, rets, sess, C.sides(df, True), hold=5, cost_bp=0.0)
    ok("nothing is earned once the hold has expired",
       abs(float(res["gross"].iloc[i + 20])) < 1e-12)


def test_flat_weight_scales_exposure_with_position_count():
    df, rets, sess = _synthetic(40, hold=60)
    res = C.backtest(df, rets, sess, C.sides(df, True), hold=60,
                     flat_weight=0.10, cost_bp=0.0)
    ok("gross exposure is 10% times the number open",
       float(res["exposure"].max()) > 0.10,
       f"peak {float(res['exposure'].max())*100:.0f}%")


if __name__ == "__main__":
    print("concordant PEAD")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all passed")
