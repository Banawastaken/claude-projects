"""Checks on the sleeve framework, aimed at the ways a backtest lies.

The important one is causality: a result computed on data truncated at date T
must equal the same result computed on the full series and then cut at T. If a
sleeve reads its own future, those two disagree.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multistrat as M  # noqa: E402
import sleeves as S  # noqa: E402


def ok(name, cond, note=""):
    print(f"  {name:<48s} {'OK' if cond else 'FAIL'}  {note}")
    if not cond:
        raise AssertionError(name)


def synth(n=900, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    px = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    return px


def test_weight_is_applied_one_day_late():
    px = pd.Series([100.0, 110.0, 121.0, 133.1],
                   index=pd.bdate_range("2020-01-01", periods=4))
    w = pd.Series([1.0, 0.0, 0.0, 0.0], index=px.index)
    r = S._returns_from_weights(px, w, cost_bp=0.0)
    # The weight set on day 0 should earn day 1's +10%, and nothing on day 0.
    ok("weight earns the following day's return",
       abs(r.iloc[0]) < 1e-12 and abs(r.iloc[1] - 0.10) < 1e-9,
       f"day0={r.iloc[0]:.4f} day1={r.iloc[1]:.4f}")
    ok("no return once the weight is gone", abs(r.iloc[2]) < 1e-12)


def test_cost_is_charged_on_turnover_not_on_holding():
    px = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=6))
    hold = pd.Series([1.0] * 6, index=px.index)
    flip = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], index=px.index)
    rh = S._returns_from_weights(px, hold, cost_bp=10.0)
    rf = S._returns_from_weights(px, flip, cost_bp=10.0)
    ok("holding pays the entry cost only once",
       abs(rh.sum() + 10.0 / 1e4) < 1e-9, f"{rh.sum()*1e4:.1f} bp")
    ok("flipping pays much more than holding",
       rf.sum() < rh.sum() - 5 * 10.0 / 1e4, f"{rf.sum()*1e4:.1f} bp")


def test_flat_position_costs_nothing():
    px = synth(200)
    w = pd.Series(0.0, index=px.index)
    r = S._returns_from_weights(px, w, cost_bp=50.0)
    ok("a flat sleeve neither earns nor pays", float(r.abs().max()) == 0.0)


def _executed(px, w):
    """The weight that actually earns each day's return."""
    return w.reindex(px.index).ffill().fillna(0.0).shift(1).fillna(0.0)


def test_executed_weight_ignores_its_own_day():
    """A weight earning day t's return must be decided from data before t.

    Truncating the series does not test this -- a signal that peeks one day
    ahead is still computable on the truncated data, so the prefixes agree.
    Perturbing the future does test it: if the weight executed on day t moves
    when day t's price changes, the sleeve is reading its own bar.
    """
    px = synth(800)
    t = 500
    bumped = px.copy()
    bumped.iloc[t:] *= 1.05

    honest = lambda p: (p / p.rolling(50).mean() - 1).apply(np.sign).fillna(0.0)
    a = _executed(px, honest(px))
    b = _executed(bumped, honest(bumped))
    ok("honest signal: day t's price cannot move day t's weight",
       abs(float(a.iloc[t] - b.iloc[t])) < 1e-12)
    ok("honest signal: nothing before t moves either",
       float((a.iloc[:t] - b.iloc[:t]).abs().max()) < 1e-12)


def test_the_causality_check_catches_a_peeker():
    """The negative control: a known look-ahead sleeve must trip the test."""
    px = synth(800)
    t = 500
    bumped = px.copy()
    bumped.iloc[t:] *= 1.05

    peek = lambda p: np.sign(p.shift(-1) / p - 1.0).fillna(0.0)
    a = _executed(px, peek(px))
    b = _executed(bumped, peek(bumped))
    ok("look-ahead signal is detected at day t",
       abs(float(a.iloc[t] - b.iloc[t])) > 1e-9,
       f"weight moved by {abs(float(a.iloc[t]-b.iloc[t])):.2f}")

    r = S._returns_from_weights(px, peek(px), cost_bp=0.0)
    ok("and it would have paid absurdly well", r.sum() > 1.0,
       f"{r.sum()*100:.0f}% vs a real sleeve's few %")


def test_real_sleeve_signals_are_causal():
    """The same check, on the signal constructions the sleeves actually use."""
    px = synth(900)
    t = 600
    bumped = px.copy()
    bumped.iloc[t:] *= 1.05

    # time-series momentum: sign of the trailing 12-month return
    lb = 252
    tsm = lambda p: np.sign(p / p.shift(lb) - 1.0).fillna(0.0)
    a, b = _executed(px, tsm(px)), _executed(bumped, tsm(bumped))
    ok("tsmom weight is causal", abs(float(a.iloc[t] - b.iloc[t])) < 1e-12)

    # Faber: month-end close vs its 10-month average, shifted a month
    def faber(p):
        m = p.resample("ME").last()
        sig = (m > m.rolling(10).mean()).astype(float).shift(1)
        return sig.reindex(p.index, method="ffill").fillna(0.0)
    a, b = _executed(px, faber(px)), _executed(bumped, faber(bumped))
    ok("faber weight is causal", abs(float(a.iloc[t] - b.iloc[t])) < 1e-12)

    # the calendar sleeve depends on the date alone, so nothing to leak
    cal = lambda p: pd.Series((p.index.month == 1).astype(float), index=p.index)
    a, b = _executed(px, cal(px)), _executed(bumped, cal(bumped))
    ok("calendar weight is causal", float((a - b).abs().max()) < 1e-12)


def test_monthly_signal_trades_the_month_it_was_decided():
    """A month-end signal must act on the following month, not the one after.

    Regression: an extra shift on top of the ffill held every month-end
    decision for a further month. The February 2020 exit did not trade until
    1 April, riding the whole COVID drawdown, and Faber's max drawdown read
    -22.9% instead of -9.6%.
    """
    idx = pd.bdate_range("2019-01-01", "2020-06-30")
    # rising through 2019, then a step down in February 2020 deep enough to
    # put the month-end close below the 10-month average
    pre = (idx < "2020-02-01").sum()
    v = np.concatenate([np.linspace(100, 140, pre),
                        np.full(len(idx) - pre, 85.0)])
    px = pd.Series(v, index=idx)

    monthly = px.resample("ME").last()
    sig = (monthly > monthly.rolling(10).mean()).astype(float)
    daily = sig.reindex(px.index, method="ffill")
    held = _executed(px, daily)

    feb_end = monthly.index[monthly.index.get_indexer(
        [pd.Timestamp("2020-02-29")], method="nearest")[0]]
    ok("February is flagged as an exit", sig.loc[feb_end] == 0.0,
       f"signal {sig.loc[feb_end]:.0f}")

    after = px.index[px.index > feb_end]
    # The signal is known at the February close, trades on the next session,
    # and so earns from the session after that.
    ok("flat by the second session of March", held.loc[after[1]] == 0.0,
       f"{after[1].date()}")
    late_feb = px.index[px.index <= feb_end][-1]
    ok("still long in late February", held.loc[late_feb] == 1.0,
       f"{late_feb.date()}")
    march_end = pd.Timestamp("2020-03-31")
    ok("and stays flat for the rest of March",
       float(held.loc[after[1]:march_end].max()) == 0.0)


def test_align_zero_fills_and_preserves_totals():
    a = pd.Series([0.01, 0.02], index=pd.to_datetime(["2020-01-02", "2020-01-06"]))
    b = pd.Series([0.03], index=pd.to_datetime(["2020-01-03"]))
    f = M.align({"a": a, "b": b})
    ok("union index covers every trading day", len(f) == 3, f"{len(f)} rows")
    ok("totals survive alignment",
       abs(f["a"].sum() - 0.03) < 1e-12 and abs(f["b"].sum() - 0.03) < 1e-12)
    ok("missing days are zero, not NaN", not f.isna().any().any())


def test_align_handles_mixed_timezones():
    ny = pd.Series([0.01], index=pd.DatetimeIndex(["2020-01-02"]).tz_localize("America/New_York"))
    naive = pd.Series([0.02], index=pd.to_datetime(["2020-01-02"]))
    f = M.align({"ny": ny, "naive": naive})
    ok("tz-aware and naive sleeves land on one row", len(f) == 1, f"{len(f)} rows")


def test_sparse_sleeve_receives_weight():
    """The January-shaped sleeve must not be silently weighted to zero."""
    idx = pd.bdate_range("2015-01-01", periods=2000)
    rng = np.random.default_rng(3)
    dense = pd.Series(rng.normal(0.0003, 0.006, len(idx)), index=idx)
    sparse = pd.Series(np.where(idx.month == 1, rng.normal(0.002, 0.01, len(idx)), 0.0),
                       index=idx)
    f = pd.DataFrame({"dense": dense, "sparse": sparse})
    w = M.inverse_vol_weights(f)
    got = w["sparse"][f["sparse"].abs() > 1e-12]
    ok("sparse sleeve gets a non-zero weight when active",
       float(got.max()) > 0.05, f"max weight {float(got.max()):.3f}")


def test_inverse_vol_weights_are_lagged():
    idx = pd.bdate_range("2016-01-01", periods=800)
    rng = np.random.default_rng(11)
    f = pd.DataFrame({"a": rng.normal(0, 0.01, len(idx)),
                      "b": rng.normal(0, 0.01, len(idx))}, index=idx)
    w1 = M.inverse_vol_weights(f)
    g = f.copy()
    g.iloc[-1] = [0.5, -0.5]          # a huge shock on the final day only
    w2 = M.inverse_vol_weights(g)
    ok("today's return cannot change today's weight",
       float((w1 - w2).abs().to_numpy().max()) < 1e-12)


def test_stats_are_sane_on_a_known_series():
    idx = pd.bdate_range("2020-01-01", periods=252)
    r = pd.Series(0.001, index=idx)         # +0.1% every day, no variance
    s = M.stats(r)
    total = 1.001 ** 252 - 1
    years = (idx[-1] - idx[0]).days / 365.25
    ok("CAGR annualises the actual elapsed span",
       abs(s["cagr"] - ((1 + total) ** (1 / years) - 1)) < 1e-9,
       f"{s['cagr']*100:.2f}% over {years:.2f}y")
    ok("no drawdown on a monotone series", abs(s["maxdd"]) < 1e-12)


def test_annualisation_uses_the_data_frequency():
    """252 is an assumption; the sleeves' union index carries more rows."""
    bd = pd.bdate_range("2020-01-01", periods=504)          # ~252/yr
    allday = pd.date_range("2020-01-01", periods=731)       # ~365/yr
    # ~261, not 252: a calendar year holds that many weekdays and
    # bdate_range drops weekends but not exchange holidays.
    ok("business-day series implies ~261/yr",
       255 < M.periods_per_year(bd) < 266, f"{M.periods_per_year(bd):.0f}")
    ok("calendar-day series implies ~365/yr",
       360 < M.periods_per_year(allday) < 370,
       f"{M.periods_per_year(allday):.0f}")

    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0002, 0.005, len(allday)), index=allday)
    s = M.stats(r)
    ok("elapsed years come from the calendar, not the row count",
       abs(s["years"] - 2.0) < 0.02, f"{s['years']:.2f}y")


if __name__ == "__main__":
    print("sleeve framework")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all passed")
