"""Order-flow feature maths, checked on synthetic books.

These run without a Databento key, so the pipeline is verified before any of
the trial credit is spent on data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mbo_features as M  # noqa: E402


def ok(name, cond, note=""):
    print(f"  {name:<50s} {'OK' if cond else 'FAIL'}  {note}")
    if not cond:
        raise AssertionError(name)


def book(rows):
    return pd.DataFrame(rows, columns=["bid_px_00", "bid_sz_00",
                                       "ask_px_00", "ask_sz_00"])


def test_size_added_at_an_unchanged_bid_is_demand():
    b = book([[100.0, 10, 100.1, 10], [100.0, 15, 100.1, 10]])
    ok("adding 5 at the bid gives +5", M.ofi(b).iloc[1] == 5.0)


def test_size_pulled_from_the_ask_is_also_demand():
    b = book([[100.0, 10, 100.1, 10], [100.0, 10, 100.1, 4]])
    ok("pulling 6 from the ask gives +6", M.ofi(b).iloc[1] == 6.0)


def test_a_rising_bid_counts_its_whole_size():
    b = book([[100.0, 10, 100.1, 10], [100.05, 7, 100.1, 10]])
    ok("bid stepping up contributes its full size", M.ofi(b).iloc[1] == 7.0)


def test_a_falling_bid_removes_the_old_size():
    b = book([[100.0, 10, 100.1, 10], [99.95, 7, 100.1, 10]])
    ok("bid stepping down contributes minus the old size",
       M.ofi(b).iloc[1] == -10.0)


def test_the_sides_are_symmetric():
    up = book([[100.0, 10, 100.1, 10], [100.0, 20, 100.1, 10]])
    dn = book([[100.0, 10, 100.1, 10], [100.0, 10, 100.1, 20]])
    ok("a bid build and an ask build are equal and opposite",
       M.ofi(up).iloc[1] == -M.ofi(dn).iloc[1], f"{M.ofi(up).iloc[1]:.0f}")


def test_first_update_has_no_imbalance():
    b = book([[100.0, 10, 100.1, 10], [100.0, 12, 100.1, 10]])
    ok("the opening row is zero, not a jump from nothing", M.ofi(b).iloc[0] == 0.0)


def test_queue_imbalance_bounds():
    b = book([[100.0, 10, 100.1, 0], [100.0, 0, 100.1, 10], [100.0, 5, 100.1, 5]])
    q = M.queue_imbalance(b)
    ok("all bid gives +1", q.iloc[0] == 1.0)
    ok("all ask gives -1", q.iloc[1] == -1.0)
    ok("balanced gives 0", q.iloc[2] == 0.0)


def test_empty_book_does_not_divide_by_zero():
    b = book([[100.0, 0, 100.1, 0]])
    ok("an empty top of book is 0, not NaN",
       float(M.queue_imbalance(b).iloc[0]) == 0.0)


def test_ofi_tracks_a_book_it_was_built_to_move():
    """A synthetic book where demand drives the mid must show it."""
    rng = np.random.default_rng(4)
    n = 4000
    push = rng.normal(0, 1, n)
    bid, ask, bs, asz = 100.0, 100.1, [], []
    bp_l, ap_l = [], []
    for k in range(n):
        step = 0.01 * np.sign(push[k]) if abs(push[k]) > 1.2 else 0.0
        bid, ask = bid + step, ask + step
        bp_l.append(bid); ap_l.append(ask)
        bs.append(max(1.0, 10 + 5 * push[k]))
        asz.append(max(1.0, 10 - 5 * push[k]))
    b = book(np.column_stack([bp_l, bs, ap_l, asz]))
    b.index = pd.date_range("2024-01-02 09:30", periods=n, freq="100ms")
    feat = M.resample_features(b, "1s")
    res = M.predictive_test(feat, horizons=(1,))
    same = float(res[res["feature"] == "ofi"]["corr_same"].iloc[0])
    ok("OFI correlates with the move it caused", same > 0.2, f"corr {same:.3f}")


if __name__ == "__main__":
    print("order-flow features")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all passed")
