"""Sanity tests for the backtest engine.

These check the properties that would silently inflate results if they were
wrong: cost accounting, breach detection, execution lag, and the tie-break when
a bar covers both the stop and the target.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import CONTRACT, Market, Rules, run_stage  # noqa: E402
from strategies import Base  # noqa: E402


def make_market(closes, spread=0.60, start="2026-01-05 08:00"):
    n = len(closes)
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    c = np.array(closes, dtype=float)
    df = pd.DataFrame({
        "ts": ts, "open": c, "high": c, "low": c, "close": c,
        "ask_open": c + spread, "ask_high": c + spread,
        "ask_low": c + spread, "ask_close": c + spread,
        "volume": np.ones(n), "minute": np.arange(n),
    })
    return Market(df)


class OneShot(Base):
    """Fire a single long at bar `at`, with fixed stop/target distances."""

    name = "test"
    sessions = ((0, 24),)
    max_trades_day = 5
    max_losses_day = 5

    def __init__(self, at=5, direction=1, sl=10.0, tp=20.0, risk=0.01):
        self.at, self.direction, self.sl, self.tp, self.risk = at, direction, sl, tp, risk
        self.fired = False
        self.reset()

    def prepare(self, mkt):
        pass

    def signal(self, i, mkt, rules, ctx):
        if i == self.at and not self.fired:
            self.fired = True
            return (self.direction, self.sl, self.tp, self.risk, "t")
        return None

    def manage(self, pos, i, mkt, rules):
        pass


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_execution_lag_and_entry_price():
    """Signal at bar i must fill at bar i+1's open, on the ask for a long."""
    mkt = make_market([2000.0] * 30)
    r = Rules(slip_entry=0.05, slip_stop=0.0, commission_per_lot=0.0)
    strat = OneShot(at=5)
    st = run_stage(mkt, strat, r, 0, 30, "funded", 6000.0, None, 6000.0)
    tr = st.trades[0]
    assert tr.idx_in == 6, f"entry bar {tr.idx_in}, expected 6"
    expected = 2000.0 + 0.60 + 0.05  # ask + entry slippage
    assert approx(tr.entry, expected), f"entry {tr.entry}, expected {expected}"
    print("  execution lag + ask-side entry           OK")


def test_round_trip_cost():
    """A flat market must lose exactly spread + slippage + commission."""
    mkt = make_market([2000.0] * 60)
    r = Rules(slip_entry=0.05, slip_stop=0.10, commission_per_lot=7.0, max_spread=5)
    strat = OneShot(at=5, sl=1.0, tp=1000.0)  # stop 1.0 away, never reached in flat tape
    st = run_stage(mkt, strat, r, 0, 60, "funded", 6000.0, None, 6000.0)
    tr = st.trades[0]
    lots = tr.lots
    # exit at final close on the bid; entry was on the ask
    expected = (2000.0 - (2000.0 + 0.60 + 0.05)) * CONTRACT * lots - 7.0 * lots
    assert approx(tr.pnl, expected, 1e-6), f"pnl {tr.pnl}, expected {expected}"
    assert tr.pnl < 0, "a flat market round trip must lose money"
    print(f"  round-trip cost accounting               OK  (${tr.pnl:.2f} on {lots} lots)")


def test_stop_wins_ties():
    """When one bar spans both stop and target, the stop must fill."""
    n = 20
    c = np.full(n, 2000.0)
    ts = pd.date_range("2026-01-05 08:00", periods=n, freq="1min", tz="UTC")
    spread = 0.6
    df = pd.DataFrame({
        "ts": ts, "open": c, "high": c, "low": c, "close": c,
        "ask_open": c + spread, "ask_high": c + spread,
        "ask_low": c + spread, "ask_close": c + spread,
        "volume": np.ones(n), "minute": np.arange(n),
    })
    # bar 8 sweeps through both the stop and the target
    df.loc[8, "high"] = 2025.0
    df.loc[8, "low"] = 1988.0
    df.loc[8, "ask_high"] = 2025.0 + spread
    df.loc[8, "ask_low"] = 1988.0 + spread
    mkt = Market(df)
    r = Rules(slip_entry=0.0, slip_stop=0.0, commission_per_lot=0.0, max_spread=5)
    strat = OneShot(at=5, sl=10.0, tp=20.0, risk=0.005)
    st = run_stage(mkt, strat, r, 0, n, "funded", 6000.0, None, 6000.0)
    tr = st.trades[0]
    assert tr.reason == "sl", f"exit reason {tr.reason}, expected sl"
    print("  stop wins same-bar ties                  OK")


def test_daily_loss_breach():
    """Equity through the daily floor must end the stage, intrabar.

    The stop is set far enough away that the gap blows through the daily limit
    before the stop can cap the loss -- which is exactly how accounts die.
    """
    c = [2000.0] * 6 + [1700.0] * 14  # gap straight through the stop
    mkt = make_market(c)
    r = Rules(slip_entry=0.0, slip_stop=250.0, commission_per_lot=0.0,
              daily_loss=0.05, max_loss=0.10, max_spread=5)
    strat = OneShot(at=3, sl=120.0, tp=5000.0, risk=0.05)
    st = run_stage(mkt, strat, r, 0, 20, "phase1", 6000.0, 0.08, 6000.0)
    assert st.trades, "test setup: no trade was taken"
    assert st.breached, "expected a breach"
    assert "daily" in st.breach_reason or "max loss" in st.breach_reason
    print(f"  drawdown breach detection                OK  ({st.breach_reason})")


def test_stop_caps_the_breach():
    """A stop nearer than the floor must save the account, not breach it.

    Before this was fixed the engine measured equity at the bar's extreme even
    though the position had already been stopped out, and killed accounts that
    would in reality have survived.
    """
    c = [2000.0] * 6 + [1500.0] * 14  # violent move far beyond the stop
    mkt = make_market(c)
    r = Rules(slip_entry=0.0, slip_stop=0.0, commission_per_lot=0.0,
              daily_loss=0.05, max_loss=0.10, max_spread=5)
    strat = OneShot(at=3, sl=20.0, tp=5000.0, risk=0.005)
    st = run_stage(mkt, strat, r, 0, 20, "phase1", 6000.0, 0.08, 6000.0)
    assert st.trades, "test setup: no trade was taken"
    assert not st.breached, f"stop should have capped the loss, got {st.breach_reason}"
    assert st.trades[0].reason == "sl"
    print("  stop caps loss before the floor          OK")


def test_static_floor_does_not_trail():
    """With a static drawdown the floor stays at 90% of the INITIAL balance."""
    r = Rules(static_drawdown=True, max_loss=0.10, initial_balance=6000.0)
    floor = r.initial_balance * (1 - r.max_loss)
    assert approx(floor, 5400.0)
    print("  static floor at $5,400                   OK")


def test_short_pays_spread_on_exit():
    """Shorts enter on the bid and exit on the ask."""
    mkt = make_market([2000.0] * 40)
    r = Rules(slip_entry=0.0, slip_stop=0.0, commission_per_lot=0.0, max_spread=5)
    strat = OneShot(at=5, direction=-1, sl=1.0, tp=1000.0)
    st = run_stage(mkt, strat, r, 0, 40, "funded", 6000.0, None, 6000.0)
    tr = st.trades[0]
    assert approx(tr.entry, 2000.0), f"short entry {tr.entry} should be the bid"
    expected = (2000.0 - (2000.0 + 0.60)) * CONTRACT * tr.lots
    assert approx(tr.pnl, expected), f"pnl {tr.pnl}, expected {expected}"
    print("  short side quoting                       OK")


def test_position_size_matches_risk():
    """Lots must put the intended risk at the stop, within rounding."""
    mkt = make_market([2000.0] * 40)
    r = Rules(slip_entry=0.0, slip_stop=0.0, commission_per_lot=0.0, max_spread=5)
    strat = OneShot(at=5, sl=20.0, tp=100.0, risk=0.01)
    st = run_stage(mkt, strat, r, 0, 40, "funded", 6000.0, None, 6000.0)
    tr = st.trades[0]
    risk_usd = 20.0 * CONTRACT * tr.lots
    assert abs(risk_usd - 60.0) <= 20.0 * CONTRACT * 0.005 + 1e-9, f"risk {risk_usd}"
    print(f"  position sizing                          OK  ({tr.lots} lots = ${risk_usd:.2f})")


if __name__ == "__main__":
    print("engine sanity tests")
    fails = 0
    for fn in [
        test_execution_lag_and_entry_price,
        test_round_trip_cost,
        test_stop_wins_ties,
        test_daily_loss_breach,
        test_stop_caps_the_breach,
        test_static_floor_does_not_trail,
        test_short_pays_spread_on_exit,
        test_position_size_matches_risk,
    ]:
        try:
            fn()
        except AssertionError as exc:
            fails += 1
            print(f"  FAIL {fn.__name__}: {exc}")
    print("all passed" if not fails else f"{fails} FAILED")
    sys.exit(1 if fails else 0)
