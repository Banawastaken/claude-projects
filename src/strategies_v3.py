"""Third-generation candidates, plus the diversifier retests.

The development sweeps produced one dominant lesson: on gold, at this account
size, the stop has to be wide. A 1.0x-ATR stop hands roughly a quarter of every
R to spread, slippage and commission and turns a real edge into noise; a
2.5x-ATR stop hands over about 5%. Every strategy tested improved monotonically
as the stop widened, which is a cost effect rather than a fitted parameter.

These add two genuinely different triggers so the four accounts are not all
running the same signal:

* `VolBreakout` keys off displacement from the daily open, so it fires on a
  clock rather than on a channel, and is usually already in a trade before a
  Donchian channel breaks.
* `KeltnerBreak` uses a volatility band around a moving average, which fires on
  acceleration away from the mean rather than on a new extreme.
"""

from __future__ import annotations

import numpy as np

from indicators import adx, atr, ema, resample, rolling_max, rolling_min, rsi
from strategies import Base
from strategies_v2 import TrendExitMixin


def _day_open(mkt):
    day = mkt.ts.astype("datetime64[D]")
    change = np.empty(len(day), dtype=bool)
    change[0] = True
    change[1:] = day[1:] != day[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(day))
    out = np.empty(len(day))
    for s, e in zip(starts, ends):
        out[s:e] = mkt.o[s]
    return out


class VolBreakout(TrendExitMixin, Base):
    """Displacement from the daily open (volatility breakout).

    Gold spends most days either going nowhere or travelling a long way. Once it
    has displaced a set fraction of its daily ATR from the open, the day tends
    to keep extending, so this takes the move in the direction of displacement.
    """

    name = "VolBreakout (daily open displacement)"
    risk_pct = 0.0060
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((6, 19),)
    exit_hour = 21

    k_open = 0.45  # displacement from the open, in daily ATR
    atr_mult = 2.0
    tp_r = 4.0
    trail_start_r = 2.0
    trail_dist_r = 1.3

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.d1 = resample(mkt, 1440)
        self.atr_h1 = atr(self.h1, 14)
        self.atr_d1 = atr(self.d1, 14)
        self.open_px = _day_open(mkt)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1, kd = self.h1.idx_map[i], self.d1.idx_map[i]
        if k1 < 25 or kd < 20:
            return None
        a1, ad = self.atr_h1[k1], self.atr_d1[kd]
        if a1 <= 0 or ad <= 0:
            return None
        c = mkt.c[i]
        disp = (c - self.open_px[i]) / ad
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        if disp >= self.k_open:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "vb_long")
        if disp <= -self.k_open:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "vb_short")
        return None

    def force_exit(self, pos, i, mkt) -> bool:
        return mkt.hour[i] >= self.exit_hour


class KeltnerBreak(TrendExitMixin, Base):
    """Break of a volatility band around the H1 mean.

    Fires on acceleration away from the mean rather than on a new N-bar extreme,
    so it tends to trigger earlier in a move than a Donchian channel does.
    """

    name = "KeltnerBreak (H1 band expansion)"
    risk_pct = 0.0060
    max_trades_day = 2
    max_losses_day = 2
    sessions = ((1, 21),)

    ema_len = 20
    band_k = 2.0
    atr_mult = 2.5
    tp_r = 5.0
    adx_min = 15
    trail_start_r = 2.0
    trail_dist_r = 1.3

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.atr_h1 = atr(self.h1, 14)
        self.mid = ema(self.h1.c, self.ema_len)
        self.adx_h1 = adx(self.h1, 14)
        self.upper = self.mid + self.band_k * self.atr_h1
        self.lower = self.mid - self.band_k * self.atr_h1

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1 = self.h1.idx_map[i]
        if k1 < self.ema_len + 25:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0 or self.adx_h1[k1] < self.adx_min:
            return None
        c = mkt.c[i]
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        if c > self.upper[k1]:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "kelt_long")
        if c < self.lower[k1]:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "kelt_short")
        return None


class FadeV3(Base):
    """Mean-reversion diversifier, retested with the wide-stop lesson applied."""

    name = "FadeV3 (range mean reversion)"
    risk_pct = 0.0050
    max_trades_day = 2
    sessions = ((7, 18),)
    exit_hour = 20

    adx_max = 20
    z_min = 2.2
    rsi_hi = 70
    rsi_lo = 30
    atr_mult = 2.5
    tp_r = 1.5

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.m15 = resample(mkt, 15)
        self.atr_h1 = atr(self.h1, 14)
        self.adx_h1 = adx(self.h1, 14)
        self.rsi15 = rsi(self.m15.c, 14)
        self.mean15 = ema(self.m15.c, 96)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1, k15 = self.h1.idx_map[i], self.m15.idx_map[i]
        if k1 < 40 or k15 < 130:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0 or self.adx_h1[k1] > self.adx_max:
            return None
        c = mkt.c[i]
        z = (c - self.mean15[k15]) / a1
        r = self.rsi15[k15]
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        if z > self.z_min and r > self.rsi_hi:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "fade_short")
        if z < -self.z_min and r < self.rsi_lo:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "fade_long")
        return None

    def force_exit(self, pos, i, mkt) -> bool:
        return mkt.hour[i] >= self.exit_hour


CANDIDATES = [VolBreakout, KeltnerBreak, FadeV3]
