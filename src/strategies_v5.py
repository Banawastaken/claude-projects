"""Fifth-generation candidates for the fourth account.

The slot-4 audition rejected Keltner bands, displacement-from-open and the
90-bar H1 channel on rule survival. What was left was another H4 Donchian,
which would have made the fourth account a near-copy of the third. These two
use structurally different reference levels so the fourth account at least
triggers on different information.
"""

from __future__ import annotations

import numpy as np

from indicators import adx, atr, ema, resample
from strategies import Base
from strategies_v2 import TrendExitMixin


def _prev_day_levels(mkt):
    """Previous completed UTC day's high and low, per M1 bar."""
    day = mkt.ts.astype("datetime64[D]")
    change = np.empty(len(day), dtype=bool)
    change[0] = True
    change[1:] = day[1:] != day[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(day))
    hi = np.full(len(day), np.nan)
    lo = np.full(len(day), np.nan)
    prev_h = prev_l = np.nan
    for s, e in zip(starts, ends):
        hi[s:e] = prev_h
        lo[s:e] = prev_l
        prev_h = mkt.h[s:e].max()
        prev_l = mkt.l[s:e].min()
    return hi, lo


class PrevDayBreak(TrendExitMixin, Base):
    """Break of the previous day's high or low.

    A level every desk watches, and structurally different from an N-bar
    channel: it resets daily rather than rolling, so it fires on different days
    than the Donchian strategies do.
    """

    name = "PrevDayBreak (prior day H/L)"
    risk_pct = 0.0075
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((7, 19),)
    exit_hour = 21

    atr_mult = 2.5
    tp_r = 4.0
    buf_atr = 0.10
    adx_min = 15
    trail_start_r = 2.0
    trail_dist_r = 1.3

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.atr_h1 = atr(self.h1, 14)
        self.adx_h1 = adx(self.h1, 14)
        self.pdh, self.pdl = _prev_day_levels(mkt)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1 = self.h1.idx_map[i]
        if k1 < 30:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0 or self.adx_h1[k1] < self.adx_min:
            return None
        hi, lo = self.pdh[i], self.pdl[i]
        if not np.isfinite(hi) or not np.isfinite(lo):
            return None
        c = mkt.c[i]
        buf = self.buf_atr * a1
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        if c > hi + buf:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "pdh_long")
        if c < lo - buf:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "pdl_short")
        return None

    def force_exit(self, pos, i, mkt) -> bool:
        return mkt.hour[i] >= self.exit_hour


class AlignedMomentum(TrendExitMixin, Base):
    """Trade only when H1, H4 and D1 all point the same way.

    A strict regime filter rather than a level: it is flat most of the time and
    only engages when gold is in an unambiguous trend, which is where the whole
    edge lived in the development tests.
    """

    name = "AlignedMomentum (H1+H4+D1)"
    risk_pct = 0.0075
    max_trades_day = 2
    max_losses_day = 2
    sessions = ((6, 20),)

    atr_mult = 2.5
    tp_r = 4.0
    break_bars = 8
    adx_min = 20
    trail_start_r = 2.0
    trail_dist_r = 1.3

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.h4 = resample(mkt, 240)
        self.d1 = resample(mkt, 1440)
        self.atr_h1 = atr(self.h1, 14)
        self.adx_h1 = adx(self.h1, 14)
        self.e1f = ema(self.h1.c, 10)
        self.e1s = ema(self.h1.c, 30)
        self.e4f = ema(self.h4.c, 10)
        self.e4s = ema(self.h4.c, 30)
        self.ed = ema(self.d1.c, 20)
        from indicators import rolling_max, rolling_min

        self.hh = rolling_max(self.h1.h, self.break_bars)
        self.ll = rolling_min(self.h1.l, self.break_bars)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1, k4, kd = self.h1.idx_map[i], self.h4.idx_map[i], self.d1.idx_map[i]
        if k1 < 40 or k4 < 35 or kd < 25:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0 or self.adx_h1[k1] < self.adx_min:
            return None
        if not np.isfinite(self.hh[k1]):
            return None
        c = mkt.c[i]
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        up = (self.e1f[k1] > self.e1s[k1] and self.e4f[k4] > self.e4s[k4]
              and self.d1.c[kd] > self.ed[kd])
        dn = (self.e1f[k1] < self.e1s[k1] and self.e4f[k4] < self.e4s[k4]
              and self.d1.c[kd] < self.ed[kd])
        if up and c > self.hh[k1]:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "align_long")
        if dn and c < self.ll[k1]:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "align_short")
        return None


CANDIDATES = [PrevDayBreak, AlignedMomentum]
