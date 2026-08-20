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
from strategies_v4 import DonchianH4
from strategies_v5 import AlignedMomentum

# Per-trade risk was chosen per strategy on the development window by funded
# rate and, more importantly, by survival once funded. Risk is not a free
# parameter here: the H4 channel gets funded slightly more often at 1.00% but
# only 16-32% of those funded accounts were still alive at the end, against
# 82-100% at 0.75%. Passing the challenge is not the goal; keeping the account
# is.
RISK = 0.0075


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
    risk_pct = 0.0100  # best funded rate on dev with no loss of survival


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
    """Same channel logic on a slower clock, so entries spread out in time.

    Originally set to a 10-bar channel with a 6R target, which looked fine over
    a 150-day horizon (86% of funded accounts still alive). Extending the
    horizon to 205 days showed those accounts blowing up later: 53% alive at
    0.75% risk and none at all at 1.00%. It was getting funded and then giving
    it back. The 20-bar channel with a 4R target survives at 100% over the same
    horizon, so the slot uses that instead.
    """

    name = "A3 Donchian H4 Swing"
    lookback = 20
    atr_mult = 1.0
    tp_r = 4.0
    adx_min = 15
    max_trades_day = 1
    max_losses_day = 1
    trail_start_r = 2.0
    trail_dist_r = 1.5
    risk_pct = RISK


class A4_AlignedMomentum(AlignedMomentum):
    """Trades only when H1, H4 and D1 all agree. Flat most of the time.

    This slot went through three occupants. Keltner band expansion had a decent
    raw edge (PF 1.48) but passed Phase 1 in only 64% of development starts, and
    displacement-from-open never got funded at all; both were dropped.

    A slower H4 channel actually scored best of everything tested here (funded
    100%, all still alive), but it is the same signal as A3 on a slightly
    different lookback, so putting it on the fourth account would have bought
    almost no diversification. This is the only non-channel-breakout signal that
    survived the rules, so it takes the slot at a small cost in robustness.
    """

    name = "A4 Multi-Timeframe Aligned Momentum"
    atr_mult = 2.5
    tp_r = 4.0
    adx_min = 25
    break_bars = 8
    max_trades_day = 2
    max_losses_day = 2
    trail_start_r = 2.0
    trail_dist_r = 1.3
    risk_pct = RISK


FINAL = [A1_DonchianH1, A2_TrendPullback, A3_DonchianH4, A4_AlignedMomentum]
