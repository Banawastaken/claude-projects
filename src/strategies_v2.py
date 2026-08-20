"""Second-generation XAUUSD candidates.

What the first generation taught us on the 2025 development window:

* Trend continuation works on gold; the Donchian breakout was the only clean
  edge (+0.22R/trade, PF 1.49).
* Plain session breakouts lose. Gold fakes out of the Asian range constantly,
  so a breakout needs a trend alignment filter and a compression precondition
  before it is worth paying the spread for.
* Average winners came in near 1.7-1.9R against 2.0-3.0R targets, meaning the
  trail was cutting trends short. Trend entries need a later trail activation
  and a wider trail distance.

Everything here is parameterised so variants can be swept rather than
hand-tuned one at a time.
"""

from __future__ import annotations

import numpy as np

from indicators import adx, atr, ema, resample, rolling_max, rolling_min, rsi, session_range
from strategies import Base


class TrendExitMixin:
    """Let winners run: activate the trail late and keep it loose."""

    trail_start_r = 2.0
    trail_dist_r = 1.3
    be_at_r = 1.2
    be_lock_r = 0.15

    def manage(self, pos, i, mkt, rules):
        risk = pos.init_risk
        if risk <= 0:
            return
        if pos.direction > 0:
            r_mult = (mkt.c[i] - pos.entry) / risk
            if r_mult >= self.be_at_r and not pos.be_moved:
                pos.sl = max(pos.sl, pos.entry + self.be_lock_r * risk)
                pos.be_moved = True
            if r_mult >= self.trail_start_r:
                pos.sl = max(pos.sl, mkt.c[i] - self.trail_dist_r * risk)
        else:
            r_mult = (pos.entry - mkt.ac[i]) / risk
            if r_mult >= self.be_at_r and not pos.be_moved:
                pos.sl = min(pos.sl, pos.entry - self.be_lock_r * risk)
                pos.be_moved = True
            if r_mult >= self.trail_start_r:
                pos.sl = min(pos.sl, mkt.ac[i] + self.trail_dist_r * risk)


class DonchianV2(TrendExitMixin, Base):
    """Refined H1 Donchian breakout, aligned with the daily trend."""

    name = "S3v2 Donchian Trend"
    risk_pct = 0.0060
    max_trades_day = 2
    max_losses_day = 2
    sessions = ((1, 21),)

    lookback = 30
    atr_mult = 1.5
    tp_r = 4.0
    adx_min = 18
    ema_d = 20
    trend_filter = True

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.d1 = resample(mkt, 1440)
        self.atr_h1 = atr(self.h1, 14)
        self.dc_hi = rolling_max(self.h1.h, self.lookback)
        self.dc_lo = rolling_min(self.h1.l, self.lookback)
        self.adx_h1 = adx(self.h1, 14)
        self.ema_d1 = ema(self.d1.c, self.ema_d)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1, kd = self.h1.idx_map[i], self.d1.idx_map[i]
        if k1 < self.lookback + 15 or kd < self.ema_d + 5:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0 or not np.isfinite(self.dc_hi[k1]):
            return None
        if self.adx_h1[k1] < self.adx_min:
            return None
        c = mkt.c[i]
        d_up = self.d1.c[kd] > self.ema_d1[kd]
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        long_ok = (not self.trend_filter) or d_up
        short_ok = (not self.trend_filter) or (not d_up)
        if long_ok and c > self.dc_hi[k1]:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "donch_long")
        if short_ok and c < self.dc_lo[k1]:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "donch_short")
        return None


class DailyBreakout(TrendExitMixin, Base):
    """S5 - classic daily-range breakout (turtle style) on gold.

    Very few trades, long holds, aimed squarely at the multi-week legs. Weekend
    holding is permitted by the firm so positions are allowed to run.
    """

    name = "S5 Daily Donchian Swing"
    risk_pct = 0.0060
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((1, 21),)

    lookback = 20
    atr_mult = 2.0
    tp_r = 5.0
    trail_start_r = 2.0
    trail_dist_r = 1.5

    def prepare(self, mkt):
        self.d1 = resample(mkt, 1440)
        self.h1 = resample(mkt, 60)
        self.atr_d1 = atr(self.d1, 14)
        self.dc_hi = rolling_max(self.d1.h, self.lookback)
        self.dc_lo = rolling_min(self.d1.l, self.lookback)
        self.ema_d = ema(self.d1.c, 50)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        kd = self.d1.idx_map[i]
        if kd < self.lookback + 55:
            return None
        ad = self.atr_d1[kd]
        if ad <= 0 or not np.isfinite(self.dc_hi[kd]):
            return None
        c = mkt.c[i]
        sl_dist = float(self.atr_mult * ad)
        risk = self.risk_for(ctx)
        above = self.d1.c[kd] > self.ema_d[kd]
        if c > self.dc_hi[kd] and above:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "d_break_long")
        if c < self.dc_lo[kd] and not above:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "d_break_short")
        return None


class CompressionBreakout(TrendExitMixin, Base):
    """S6 - Asian-range breakout, but only after volatility compression and
    only in the direction of the daily trend.

    The v1 version lost money because it took every break of the overnight
    range. This one requires the range to be genuinely tight relative to recent
    daily ATR (compression precedes expansion) and refuses counter-trend breaks.
    """

    name = "S6 Compression Breakout (London)"
    risk_pct = 0.0060
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((7, 12),)
    exit_hour = 19

    max_width_atr = 0.50  # Asian range must be tight vs daily ATR
    min_width_atr = 0.12
    buf_atr = 0.15
    atr_mult = 0.9
    tp_r = 3.0
    trend_filter = True

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.d1 = resample(mkt, 1440)
        self.atr_h1 = atr(self.h1, 14)
        self.atr_d1 = atr(self.d1, 14)
        self.ema_d1 = ema(self.d1.c, 20)
        self.hi, self.lo = session_range(mkt, 0, 7)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        hi, lo = self.hi[i], self.lo[i]
        if not np.isfinite(hi):
            return None
        k1, kd = self.h1.idx_map[i], self.d1.idx_map[i]
        if k1 < 25 or kd < 25:
            return None
        a1, ad = self.atr_h1[k1], self.atr_d1[kd]
        if a1 <= 0 or ad <= 0:
            return None
        width = hi - lo
        if not (self.min_width_atr * ad <= width <= self.max_width_atr * ad):
            return None
        c = mkt.c[i]
        buf = self.buf_atr * a1
        sl_dist = float(np.clip(max(0.55 * width, self.atr_mult * a1), 0.7 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        d_up = self.d1.c[kd] > self.ema_d1[kd]
        long_ok = (not self.trend_filter) or d_up
        short_ok = (not self.trend_filter) or (not d_up)
        if long_ok and c > hi + buf:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "compr_long")
        if short_ok and c < lo - buf:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "compr_short")
        return None

    def force_exit(self, pos, i, mkt) -> bool:
        return mkt.hour[i] >= self.exit_hour


class PullbackV2(TrendExitMixin, Base):
    """S7 - trend pullback with a real retracement requirement.

    v1 fired on any 4-bar M15 high inside a trend, which is noise. This waits
    for price to actually trade back into the H1 EMA20 zone, then requires an
    H1 close reasserting the trend before entering.
    """

    name = "S7 Trend Pullback v2"
    risk_pct = 0.0060
    max_trades_day = 2
    sessions = ((6, 20),)

    atr_mult = 1.2
    tp_r = 3.5
    adx_min = 20
    pull_atr = 0.5  # how close to EMA20 the pullback must come
    lookback_bars = 12

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.h4 = resample(mkt, 240)
        self.ema20 = ema(self.h1.c, 20)
        self.ema50 = ema(self.h1.c, 50)
        self.ema_h4 = ema(self.h4.c, 50)
        self.atr_h1 = atr(self.h1, 14)
        self.adx_h1 = adx(self.h1, 14)
        # did price visit the EMA20 zone within the last N H1 bars?
        n = self.h1.n
        dist = np.abs(self.h1.l - self.ema20)
        dist_up = np.abs(self.h1.h - self.ema20)
        self.touch_long = np.zeros(n, dtype=bool)
        self.touch_short = np.zeros(n, dtype=bool)
        atr_safe = np.where(self.atr_h1 <= 0, 1e-9, self.atr_h1)
        near_l = (self.h1.l - self.ema20) / atr_safe
        near_h = (self.h1.h - self.ema20) / atr_safe
        vis_l = near_l <= self.pull_atr
        vis_h = near_h >= -self.pull_atr
        for k in range(self.lookback_bars, n):
            self.touch_long[k] = vis_l[k - self.lookback_bars : k].any()
            self.touch_short[k] = vis_h[k - self.lookback_bars : k].any()

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1, k4 = self.h1.idx_map[i], self.h4.idx_map[i]
        if k1 < 70 or k4 < 55:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0 or self.adx_h1[k1] < self.adx_min:
            return None
        e20, e50, e4 = self.ema20[k1], self.ema50[k1], self.ema_h4[k4]
        c1, c4 = self.h1.c[k1], self.h4.c[k4]
        c = mkt.c[i]
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        up = e20 > e50 and c1 > e20 and c4 > e4
        dn = e20 < e50 and c1 < e20 and c4 < e4
        if up and self.touch_long[k1] and c > self.h1.h[k1]:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "pb2_long")
        if dn and self.touch_short[k1] and c < self.h1.l[k1]:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "pb2_short")
        return None


class MomentumDay(TrendExitMixin, Base):
    """S8 - New York continuation of an already-directional day.

    If gold has already travelled a meaningful fraction of its daily ATR in one
    direction by the New York open, that direction tends to persist into the
    close. Entry is a break of the pre-NY session extreme.
    """

    name = "S8 NY Momentum Continuation"
    risk_pct = 0.0060
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((13, 18),)
    exit_hour = 20

    min_travel = 0.35  # of daily ATR, measured 00:00->13:00
    atr_mult = 1.1
    tp_r = 2.5

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.d1 = resample(mkt, 1440)
        self.atr_h1 = atr(self.h1, 14)
        self.atr_d1 = atr(self.d1, 14)
        self.hi, self.lo = session_range(mkt, 0, 13)
        self.open_px = self._day_open(mkt)

    @staticmethod
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

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        hi, lo = self.hi[i], self.lo[i]
        if not np.isfinite(hi):
            return None
        k1, kd = self.h1.idx_map[i], self.d1.idx_map[i]
        if k1 < 25 or kd < 20:
            return None
        a1, ad = self.atr_h1[k1], self.atr_d1[kd]
        if a1 <= 0 or ad <= 0:
            return None
        c = mkt.c[i]
        o = self.open_px[i]
        travel = (c - o) / ad
        sl_dist = float(np.clip(self.atr_mult * a1, 0.8 * a1, 4.0 * a1))
        risk = self.risk_for(ctx)
        if travel >= self.min_travel and c > hi:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "nymom_long")
        if travel <= -self.min_travel and c < lo:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "nymom_short")
        return None

    def force_exit(self, pos, i, mkt) -> bool:
        return mkt.hour[i] >= self.exit_hour


class FadeExtreme(Base):
    """S9 - fade a stretched move back toward the mean, range regimes only.

    Kept as a decorrelation candidate: it is the only non-trend logic in the
    pool, so it earns its place if it can stay flat-to-positive while the trend
    strategies drawdown.
    """

    name = "S9 Range Fade v2"
    risk_pct = 0.0050
    max_trades_day = 2
    sessions = ((7, 18),)
    exit_hour = 20

    adx_max = 20
    z_min = 2.2
    rsi_hi = 72
    rsi_lo = 28
    atr_mult = 1.3
    tp_r = 1.6

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


CANDIDATES = [
    DonchianV2,
    DailyBreakout,
    CompressionBreakout,
    PullbackV2,
    MomentumDay,
    FadeExtreme,
]
