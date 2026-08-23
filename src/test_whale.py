"""Checks on the block-print framework, aimed at the bug that broke it.

`forward` reads the exit price from the full minute table while measuring a
sparse selection. Getting that wrong does not raise -- it silently drops almost
every observation, which reads as "too few trades" rather than as an error.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import whale as W  # noqa: E402


def ok(name, cond, note=""):
    print(f"  {name:<52s} {'OK' if cond else 'FAIL'}  {note}")
    if not cond:
        raise AssertionError(name)


def session_bars(n=120, start="2026-04-01 14:00", step="1min", px0=20000.0):
    idx = pd.date_range(start, periods=n, freq=step, tz="UTC")
    return pd.DataFrame({
        "close": px0 + np.arange(n) * 1.0,          # +1 point a minute
        "print_side": 1.0, "print_size": 10.0,
    }, index=idx)


def test_forward_reads_prices_from_the_full_table():
    bar = session_bars()
    sparse = bar.iloc[::30]                          # every 30th minute
    good = W.forward(sparse, 20, bar)
    bad = W.forward(sparse, 20)                      # the bug: sparse lookup
    ok("sparse selection keeps its observations",
       good.notna().sum() >= 3, f"{int(good.notna().sum())} of {len(sparse)}")
    ok("looking up inside the selection loses them",
       bad.notna().sum() < good.notna().sum(),
       f"{int(bad.notna().sum())} vs {int(good.notna().sum())}")


def test_forward_measures_clock_time_not_rows():
    bar = session_bars()
    sparse = bar.iloc[::30]
    v = W.forward(sparse, 20, bar).dropna()
    # +1 point a minute, long side, 20 minutes -> +20 points every time.
    ok("20 minutes of a 1pt/min drift is 20 points",
       np.allclose(v.to_numpy(), 20.0), f"{v.iloc[0]:.1f}")


def test_short_side_flips_the_sign():
    bar = session_bars()
    bar["print_side"] = -1.0
    v = W.forward(bar.iloc[::30], 20, bar).dropna()
    ok("a short print in a rising market loses",
       np.allclose(v.to_numpy(), -20.0), f"{v.iloc[0]:.1f}")


def test_window_may_not_leave_the_session():
    bar = session_bars(n=400, start="2026-04-01 17:30")   # runs past 20:00 UTC
    v = W.forward(bar, 20, bar)
    last_ok = v.dropna().index.max()
    ok("no window ends after the regular session closes",
       last_ok + pd.Timedelta(minutes=20) <= last_ok.normalize() + pd.Timedelta(hours=20),
       f"last entry {last_ok.strftime('%H:%M')}")


def test_window_may_not_span_a_contract_roll():
    """The failure that produced $2,518 a trade before it was caught."""
    a = session_bars(n=60, start="2026-04-01 14:00", px0=20000.0)
    b = session_bars(n=60, start="2026-04-02 14:00", px0=20500.0)  # +500 gap
    bar = pd.concat([a, b])
    v = W.forward(bar, 20, bar).dropna()
    ok("no observation carries the overnight gap",
       float(v.abs().max()) < 100.0, f"max {float(v.abs().max()):.1f} pts")


def test_exit_on_an_untraded_minute_is_dropped():
    bar = session_bars()
    holed = bar.drop(bar.index[40])
    v = W.forward(bar.iloc[[20]], 20, holed)
    ok("a missing exit minute is dropped, not approximated",
       bool(v.isna().all()))


def test_summarise_profit_factor():
    idx = pd.date_range("2026-04-01 14:00", periods=100, freq="1min", tz="UTC")
    p = pd.Series([2.0] * 60 + [-1.0] * 40, index=idx)
    s = W.summarise(p, 20.0, "x", cost_ticks=0.0)
    ok("profit factor is wins over losses",
       abs(s["pf"] - (60 * 2.0) / (40 * 1.0)) < 1e-9, f"{s['pf']:.2f}")
    ok("hit rate counts winners", abs(s["hit"] - 0.6) < 1e-9)


def test_cost_is_charged_per_trade():
    idx = pd.date_range("2026-04-01 14:00", periods=50, freq="1min", tz="UTC")
    p = pd.Series(1.0, index=idx)
    free = W.summarise(p, 20.0, "x", cost_ticks=0.0)
    paid = W.summarise(p, 20.0, "x", cost_ticks=1.0, tick=0.25)
    ok("one tick of cost is deducted from every trade",
       abs((free["net"] - paid["net"]) - 0.25 * 20.0) < 1e-9,
       f"{free['net'] - paid['net']:.2f}")


if __name__ == "__main__":
    print("block-print framework")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all passed")
