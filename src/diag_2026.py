"""Is the 'only trend-following works' conclusion still true in 2026?

That conclusion came from 2025 data, which was a clean uptrend. 2026 is a
different animal: daily ATR nearly doubled and the market swung both ways. If
mean reversion works in that regime, the whole strategy set is built on a
premise that expired.

This is diagnostic only. Anything it changes about the final strategies makes
those strategies partly fitted to the test window, and the report says so.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market  # noqa: E402
from evaluate import raw_edge  # noqa: E402
from run import load, slice_period  # noqa: E402

import strategies_final as F  # noqa: E402
import strategies_v3 as V3  # noqa: E402
import strategies_v4 as V4  # noqa: E402


def variant(base, **params):
    return type(base.__name__ + "_v", (base,), {**params, "name": base.name})


PROBES = [
    ("TREND  A1 Donchian H1", F.A1_DonchianH1, {}),
    ("TREND  A2 Pullback", F.A2_TrendPullback, {}),
    ("TREND  A3 Donchian H4", F.A3_DonchianH4, {}),
    ("TREND  A4 Aligned", F.A4_AlignedMomentum, {}),
    ("REVERT FadeV3 atr2.5 tp1.5", V3.FadeV3, {"atr_mult": 2.5, "tp_r": 1.5}),
    ("REVERT FadeV3 atr2.5 tp2.5", V3.FadeV3, {"atr_mult": 2.5, "tp_r": 2.5}),
    ("REVERT FadeV3 wide adx30", V3.FadeV3, {"atr_mult": 2.5, "tp_r": 2.0, "adx_max": 30}),
    ("REVERT FalseBreak lb20", V4.FalseBreak, {"lookback": 20, "atr_mult": 2.0, "tp_r": 3.0}),
    ("REVERT FalseBreak lb30 t2", V4.FalseBreak, {"lookback": 30, "atr_mult": 2.5, "tp_r": 2.0}),
]

if __name__ == "__main__":
    df = load("xauusd_m1_clean")
    mkt = Market(df)
    windows = [
        ("DEV  2025-02 .. 2025-12", "2025-02-01", "2025-12-01"),
        ("TEST 2025-12 .. 2026-08", "2025-12-01", None),
        ("CHOP 2026-04 .. 2026-08", "2026-04-01", None),
    ]
    for label, a, b in windows:
        i0, i1 = slice_period(df, a, b)
        print(f"\n=== {label}")
        print(f"{'probe':30s} {'trades':>7s} {'WR%':>6s} {'expR':>7s} {'totR':>7s} {'PF':>6s}")
        for name, cls, params in PROBES:
            c = variant(cls, **params) if params else cls
            e = raw_edge(c(), mkt, i0, i1)
            print(f"{name:30s} {e['trades']:7d} {e['win_rate'] * 100:6.1f} "
                  f"{e['expectancy_R']:7.3f} {e['total_R']:7.1f} {e['pf']:6.2f}", flush=True)
