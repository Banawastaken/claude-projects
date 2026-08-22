"""Checks on the gamma engine, aimed at the parts that can be silently wrong."""

from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gex  # noqa: E402


def chain(strikes, rights, ois, dte=7, iv=0.20):
    n = len(strikes)
    return {"strike": np.asarray(strikes, float),
            "right": np.asarray(rights, float),
            "oi": np.asarray(ois, float),
            "iv": np.full(n, iv),
            "dte": np.full(n, float(dte)),
            "gamma_quoted": np.zeros(n),
            "volume": np.zeros(n)}


def ok(name, cond, note=""):
    print(f"  {name:<44s} {'OK' if cond else 'FAIL'}  {note}")
    if not cond:
        raise AssertionError(name)


def test_bs_gamma_peaks_at_the_money():
    ks = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    g = gex.bs_gamma(100.0, ks, 30, np.full(5, 0.2))
    ok("BS gamma peaks near the money", int(np.argmax(g)) == 2,
       f"peak at K={ks[np.argmax(g)]:.0f}")


def test_gamma_rises_as_expiry_nears():
    g_far = gex.bs_gamma(100.0, np.array([100.0]), 30, np.array([0.2]))[0]
    g_near = gex.bs_gamma(100.0, np.array([100.0]), 1, np.array([0.2]))[0]
    ok("ATM gamma rises into expiry", g_near > g_far,
       f"{g_far:.4f} -> {g_near:.4f}")


def test_zero_dte_is_finite():
    g = gex.bs_gamma(100.0, np.array([100.0]), 0, np.array([0.2]))[0]
    ok("0DTE gamma is large but finite", np.isfinite(g) and g > 0, f"{g:.2f}")


def test_dealer_sign_convention():
    """Default convention: long calls positive, short puts negative."""
    c = chain([100.0], [1], [1000.0])
    p = chain([100.0], [-1], [1000.0])
    vc = gex.gex_at(100.0, c)[0].sum()
    vp = gex.gex_at(100.0, p)[0].sum()
    ok("calls add positive dealer gamma", vc > 0, f"{vc:,.0f}")
    ok("puts subtract dealer gamma", vp < 0, f"{vp:,.0f}")
    flipped = gex.gex_at(100.0, c, dealer_sign=(-1.0, 1.0))[0].sum()
    ok("dealer_sign flips the sign", math.isclose(flipped, -vc, rel_tol=1e-9))


def test_flip_sits_between_put_and_call_strikes():
    """Puts below, calls above -> short gamma low, long gamma high."""
    ch = chain([90.0, 110.0], [-1, 1], [5000.0, 5000.0])
    grid, tot = gex.profile(ch, 100.0, 0.85, 1.15, 301)
    flip = gex.gamma_flip(grid, tot)
    ok("flip found between the two strikes",
       flip is not None and 90.0 < flip < 110.0, f"flip={flip:.2f}")
    below = gex.gex_at(flip - 5, ch)[0].sum()
    above = gex.gex_at(flip + 5, ch)[0].sum()
    ok("short gamma below flip, long above", below < 0 < above,
       f"{below:,.0f} / {above:,.0f}")


def test_no_flip_when_profile_never_crosses():
    ch = chain([100.0], [1], [1000.0])  # calls only: positive everywhere
    grid, tot = gex.profile(ch, 100.0)
    ok("no flip reported when sign never changes",
       gex.gamma_flip(grid, tot) is None)


def test_max_dte_filter_excludes_contracts():
    ch = chain([100.0, 100.0], [1, 1], [1000.0, 1000.0])
    ch["dte"] = np.array([1.0, 60.0])
    all_v = gex.gex_at(100.0, ch)[0].sum()
    near = gex.gex_at(100.0, ch, max_dte=5)[0].sum()
    ok("max_dte drops far-dated contracts", 0 < near < all_v,
       f"{near:,.0f} of {all_v:,.0f}")


def test_scales_with_open_interest():
    a = gex.gex_at(100.0, chain([100.0], [1], [1000.0]))[0].sum()
    b = gex.gex_at(100.0, chain([100.0], [1], [2000.0]))[0].sum()
    ok("GEX is linear in open interest", math.isclose(b, 2 * a, rel_tol=1e-9))


def test_parses_osi_symbols():
    payload = {"timestamp": "2026-08-22 20:30:23",
               "data": {"current_price": 100.0, "options": [
                   {"option": "NDX260918C04000000", "open_interest": 5,
                    "iv": 0.2, "gamma": 0.01, "volume": 0},
                   {"option": "NDX260918P04500000", "open_interest": 7,
                    "iv": 0.3, "gamma": 0.02, "volume": 0},
                   {"option": "NDX260918C05000000", "open_interest": 0,
                    "iv": 0.2, "gamma": 0.01, "volume": 0}]}}
    spot, ch = gex.parse_chain(payload)
    ok("strike decoded from OSI", list(ch["strike"]) == [4000.0, 4500.0],
       f"{list(ch['strike'])}")
    ok("call/put decoded", list(ch["right"]) == [1.0, -1.0])
    ok("zero-OI contracts dropped", len(ch["strike"]) == 2)
    ok("spot read from payload", spot == 100.0)


if __name__ == "__main__":
    print("gamma engine")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("all passed")
