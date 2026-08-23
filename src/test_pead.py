"""Checks on the PEAD pipeline, aimed at the two places look-ahead hides.

An 8-K accepted after the close is not information for that day, and ranking an
event against the full sample would let it see its own future. Both are tested
here, each with a negative control that must fail if the check is toothless.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pead as P  # noqa: E402


def ok(name, cond, note=""):
    print(f"  {name:<52s} {'OK' if cond else 'FAIL'}  {note}")
    if not cond:
        raise AssertionError(name)


SESSIONS = pd.DatetimeIndex(pd.bdate_range("2024-01-02", periods=30))


def test_after_close_release_trades_the_next_session():
    ev = {"date": "2024-01-10", "accepted": "2024-01-10T21:30:00.000Z"}
    i = P.reaction_day(ev, SESSIONS)
    ok("release accepted 16:30 ET reacts the next session",
       SESSIONS[i] == pd.Timestamp("2024-01-11"), f"{SESSIONS[i].date()}")


def test_before_close_release_trades_the_same_session():
    ev = {"date": "2024-01-10", "accepted": "2024-01-10T13:05:00.000Z"}
    i = P.reaction_day(ev, SESSIONS)
    ok("release accepted 08:05 ET reacts the same session",
       SESSIONS[i] == pd.Timestamp("2024-01-10"), f"{SESSIONS[i].date()}")


def test_release_with_no_timestamp_is_treated_as_after_close():
    ev = {"date": "2024-01-10", "accepted": ""}
    i = P.reaction_day(ev, SESSIONS)
    ok("unknown acceptance time defers to the next session",
       SESSIONS[i] == pd.Timestamp("2024-01-11"),
       "conservative: delaying entry can never manufacture an edge")


def test_friday_evening_release_trades_monday():
    ev = {"date": "2024-01-12", "accepted": "2024-01-12T22:00:00.000Z"}
    i = P.reaction_day(ev, SESSIONS)
    ok("Friday-evening release reacts on Monday",
       SESSIONS[i] == pd.Timestamp("2024-01-15"), f"{SESSIONS[i].date()}")


def _frame(n=600, seed=1):
    rng = np.random.default_rng(seed)
    react = np.sort(rng.integers(0, 4000, n))
    return pd.DataFrame({
        "ticker": [f"T{i%40}" for i in range(n)],
        "react_i": react, "entry_i": react + 2,
        "reaction": rng.normal(0, 0.05, n), "adv": 1e8,
    })


def test_percentile_ignores_events_that_had_not_happened():
    df = _frame()
    full = P.causal_percentile(df)
    cut = 400
    part = P.causal_percentile(df.iloc[:cut].reset_index(drop=True))
    a = full.iloc[:cut].to_numpy()
    b = part.to_numpy()
    both = ~(np.isnan(a) | np.isnan(b))
    ok("scores are unchanged when later events are removed",
       both.sum() > 100 and np.nanmax(np.abs(a[both] - b[both])) < 1e-12,
       f"{both.sum()} comparable scores")


def test_percentile_would_catch_a_full_sample_ranking():
    """Negative control: ranking against everything must trip the same test."""
    df = _frame()
    peek = df["reaction"].rank(pct=True)          # uses the whole sample
    cut = 400
    peek_part = df.iloc[:cut]["reaction"].rank(pct=True)
    diff = float(np.abs(peek.iloc[:cut].to_numpy() - peek_part.to_numpy()).max())
    ok("a whole-sample ranking is detected as non-causal", diff > 1e-6,
       f"max shift {diff:.3f}")


def test_percentile_needs_history_before_it_scores():
    df = _frame()
    p = P.causal_percentile(df, min_history=200)
    ok("earliest events are left unscored", p.iloc[:150].isna().all())
    ok("later events do get scored", p.iloc[-50:].notna().any())


def test_percentile_orders_reactions_correctly():
    df = _frame(400, seed=7)
    df.loc[399, "reaction"] = df["reaction"].max() + 1.0
    p = P.causal_percentile(df, min_history=50)
    ok("the largest reaction scores at the top",
       p.iloc[399] > 0.99, f"{p.iloc[399]:.3f}")


def test_costs_rise_as_liquidity_falls():
    tiers = [P.cost_bp(v) for v in (5e8, 5e7, 1e7, 1e5)]
    ok("cost is monotone in illiquidity",
       all(tiers[i] < tiers[i+1] for i in range(len(tiers)-1)), f"{tiers}")
    ok("the illiquid tier is genuinely punitive", tiers[-1] >= 50,
       f"{tiers[-1]:.0f} bp round trip")


if __name__ == "__main__":
    print("PEAD pipeline")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all passed")
