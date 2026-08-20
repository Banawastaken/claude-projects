"""The four strategies selected on the 2025 development window.

Selection process, for the record:

Ten distinct concepts were tested. Every counter-trend concept lost money at
every parameter setting tried -- the mean-reversion fade (-0.13 to -0.19 R),
the failed-breakout reversal (-0.11 to -0.24 R) and the plain Asian-range
compression breakout (-0.03 to -0.23 R) were all discarded rather than tuned
until they looked profitable. On gold over this period only trend continuation
paid, so all four survivors are trend-following. That is a real and uncomfortable
result: these four accounts will win and lose together, and the report says so.

Each config sits in the middle of a broad plateau found by sweeping, not on a
single best cell. The dominant parameter everywhere was stop width, which is a
cost effect: a round trip costs roughly $0.72 in price terms, so a $3 stop
surrenders 24% of every R while a $25 stop surrenders 3%.
"""

from __future__ import annotations

from strategies_v2 import DonchianV2, PullbackV2
from strategies_v3 import KeltnerBreak
from strategies_v4 import DonchianH4

RISK = 0.0075  # per-trade risk used by all four unless overridden


class A1_DonchianH1(DonchianV2):
    """H1 channel breakout. The core trend engine."""

    name = "A1 Donchian H1 Breakout"
    lookback = 45
    atr_mult = 2.5
    tp_r = 5.0
    adx_min = 18
    trend_filter = False
    max_trades_day = 2
    max_losses_day = 2
    trail_start_r = 2.0
    trail_dist_r = 1.3
    risk_pct = RISK


class A2_TrendPullback(PullbackV2):
    """Buys the pullback inside an H1/H4 trend instead of the breakout."""

    name = "A2 H1 Trend Pullback"
    atr_mult = 2.0
    tp_r = 2.5
    adx_min = 20
    max_trades_day = 2
    max_losses_day = 2
    trail_start_r = 2.0
    trail_dist_r = 1.3
    risk_pct = RISK


class A3_DonchianH4(DonchianH4):
    """Same channel logic on a slower clock, so entries spread out in time."""

    name = "A3 Donchian H4 Swing"
    lookback = 10
    atr_mult = 1.0
    tp_r = 6.0
    adx_min = 15
    max_trades_day = 1
    max_losses_day = 1
    trail_start_r = 2.0
    trail_dist_r = 1.5
    risk_pct = RISK


class A4_Keltner(KeltnerBreak):
    """Volatility-band expansion: fires on acceleration, not on a new extreme."""

    name = "A4 Keltner Band Expansion"
    band_k = 2.5
    atr_mult = 3.0
    tp_r = 5.0
    adx_min = 0
    max_trades_day = 2
    max_losses_day = 2
    trail_start_r = 2.0
    trail_dist_r = 1.3
    risk_pct = RISK


FINAL = [A1_DonchianH1, A2_TrendPullback, A3_DonchianH4, A4_Keltner]
