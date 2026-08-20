"""Fourth-generation candidates, invented to replace the two that failed.

The mean-reversion fade and the compression breakout were negative at every
parameter setting tested on the development window, so both were discarded
rather than tuned until they looked profitable. These are their replacements.

`FalseBreak` is the deliberate decorrelation play: it only triggers when a
breakout fails, so it is structurally anti-correlated with the three breakout
strategies. `DonchianH4` trades the same edge as the H1 Donchian but on a
slower clock, which spreads entries out in time.
"""

from __future__ import annotations

import numpy as np

from indicators import adx, atr, ema, resample, rolling_max, rolling_min
from strategies import Base
from strategies_v2 import TrendExitMixin


class FalseBreak(TrendExitMixin, Base):
    """Fade a breakout that fails and closes back inside the range.

    Price pokes through an N-bar extreme, fails to hold, and closes back inside
    the channel. That trapped-trader pattern tends to run to the opposite side
    of the range, and by construction it fires on exactly the days the breakout
    strategies get chopped up.
    """

    name = "FalseBreak (failed breakout reversal)"
    risk_pct = 0.0055
    max_trades_day = 2
    max_losses_day = 2
    sessions = ((6, 20),)

    lookback = 30
    atr_mult = 2.0
    tp_r = 3.0
    confirm_bars = 3  # H1 bars allowed for the break to fail
    trail_start_r = 2.0
    trail_dist_r = 1.3

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.atr_h1 = atr(self.h1, 14)
        self.dc_hi = rolling_max(self.h1.h, self.lookback)
        self.dc_lo = rolling_min(self.h1.l, self.lookback)
        n = self.h1.n
        # A failed upside break: some bar in the window poked above the channel
        # but the most recent close is back inside it.
        poked_up = np.zeros(n, dtype=bool)
        poked_dn = np.zeros(n, dtype=bool)
        hi_ok = np.isfinite(self.dc_hi)
        for k in range(self.lookback + 2, n):
            w = slice(max(0, k - self.confirm_bars), k + 1)
            if hi_ok[k]:
                poked_up[k] = np.any(self.h1.h[w] > self.dc_hi[w])
                poked_dn[k] = np.any(self.h1.l[w] < self.dc_lo[w])
        self.poked_up = poked_up
        self.poked_dn = poked_dn

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1 = self.h1.idx_map[i]
        if k1 < self.lookback + 20:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0 or not np.isfinite(self.dc_hi[k1]):
            return None
        c = mkt.c[i]
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        # failed upside break -> short back into the range
        if self.poked_up[k1] and self.h1.c[k1] < self.dc_hi[k1] and c < self.h1.l[k1]:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "fb_short")
        if self.poked_dn[k1] and self.h1.c[k1] > self.dc_lo[k1] and c > self.h1.h[k1]:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "fb_long")
        return None


class DonchianH4(TrendExitMixin, Base):
    """The Donchian edge on a slower clock (H4 channel, H4 ATR stops)."""

    name = "DonchianH4 (slow channel)"
    risk_pct = 0.0060
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((1, 21),)

    lookback = 20
    atr_mult = 1.5
    tp_r = 5.0
    adx_min = 15
    trail_start_r = 2.0
    trail_dist_r = 1.5

    def prepare(self, mkt):
        self.h4 = resample(mkt, 240)
        self.atr_h4 = atr(self.h4, 14)
        self.dc_hi = rolling_max(self.h4.h, self.lookback)
        self.dc_lo = rolling_min(self.h4.l, self.lookback)
        self.adx_h4 = adx(self.h4, 14)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k4 = self.h4.idx_map[i]
        if k4 < self.lookback + 20:
            return None
        a4 = self.atr_h4[k4]
        if a4 <= 0 or not np.isfinite(self.dc_hi[k4]):
            return None
        if self.adx_h4[k4] < self.adx_min:
            return None
        c = mkt.c[i]
        sl_dist = float(np.clip(self.atr_mult * a4, 0.5 * a4, 3.0 * a4))
        risk = self.risk_for(ctx)
        if c > self.dc_hi[k4]:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "d4_long")
        if c < self.dc_lo[k4]:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "d4_short")
        return None


CANDIDATES = [FalseBreak, DonchianH4]
