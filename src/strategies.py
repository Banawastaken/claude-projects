"""XAUUSD strategies built for prop-firm survival rather than raw return.

Every strategy shares the same risk governor in `Base`, because on a $6K
account with a 5% daily and 10% static max loss, the thing that fails a
challenge is almost never a weak edge -- it is one bad day. The governor
enforces a per-day loss stop, a trade cap, de-risking after consecutive
losses, and de-risking as the account nears the max-loss floor.

Cost note that drives the design: the cost of a round trip is roughly
(spread + slippage + commission) in price units, about $0.72 on gold here.
As a fraction of risk that is (0.72 / stop_distance), so a $3 stop burns 24%
of every R while a $15 stop burns under 5%. All of these use ATR-scaled stops
that stay wide enough to keep costs a small share of the edge.
"""

from __future__ import annotations

import numpy as np

from indicators import TF, adx, atr, ema, resample, rolling_max, rolling_min, rsi, session_range


class Base:
    name = "base"
    risk_pct = 0.0060
    max_trades_day = 2
    max_losses_day = 2  # two stop-outs and the day is over
    daily_stop_pct = 0.020  # halt for the day after this much loss
    max_consec_losses = 4
    sessions = ((7, 20),)  # UTC windows in which entries are allowed

    def prepare(self, mkt):
        raise NotImplementedError

    def reset(self):
        self._cooloff = False
        self._last_consec = 0

    def on_new_day(self, i, mkt):
        self._cooloff = False  # a cool-off lasts to the end of the day only

    def force_exit(self, pos, i, mkt) -> bool:
        return False

    # -- shared risk governance ------------------------------------------
    def in_session(self, i, mkt) -> bool:
        h = mkt.hour[i]
        return any(a <= h < b for a, b in self.sessions)

    def allowed(self, i, mkt, rules, ctx) -> bool:
        if mkt.dow[i] >= 5:
            return False
        if ctx.trades_today >= self.max_trades_day:
            return False
        if ctx.losses_today >= self.max_losses_day:
            return False
        if ctx.day_pnl_pct <= -self.daily_stop_pct:
            return False
        # A run of losses stands the strategy down for the rest of the day,
        # rather than banning it outright.
        if ctx.consec_losses >= self.max_consec_losses:
            if ctx.consec_losses > self._last_consec:
                self._cooloff = True
                self._last_consec = ctx.consec_losses
        else:
            self._last_consec = ctx.consec_losses
        if self._cooloff:
            return False
        if mkt.spread[i] > rules.max_spread:
            return False
        return self.in_session(i, mkt)

    def risk_for(self, ctx) -> float:
        r = self.risk_pct
        # Target already banked and only the minimum-trading-days requirement
        # left: trade token size rather than risk the pass to tick a box.
        if ctx.target_balance and ctx.balance >= ctx.target_balance:
            return 0.0015
        if ctx.consec_losses >= 2:
            r *= 0.60  # shrink after a bad run instead of pressing
        # As equity approaches the static floor, cut size hard so a losing
        # streak decays toward the floor instead of hitting it.
        cushion = (ctx.balance - ctx.floor_equity) / (ctx.day_start_balance * 0.10)
        if cushion < 0.55:
            r *= 0.60
        if cushion < 0.35:
            r *= 0.50
        return max(r, 0.0015)

    def manage(self, pos, i, mkt, rules):
        """Default: break-even at +1R, then trail at 1.5R behind the extreme."""
        risk = pos.init_risk
        if risk <= 0:
            return
        if pos.direction > 0:
            gain = mkt.c[i] - pos.entry
            r_mult = gain / risk
            if r_mult >= 1.0 and not pos.be_moved:
                pos.sl = max(pos.sl, pos.entry + 0.10)
                pos.be_moved = True
            if r_mult >= 1.8:
                pos.sl = max(pos.sl, mkt.c[i] - 0.9 * risk)
        else:
            gain = pos.entry - mkt.ac[i]
            r_mult = gain / risk
            if r_mult >= 1.0 and not pos.be_moved:
                pos.sl = min(pos.sl, pos.entry - 0.10)
                pos.be_moved = True
            if r_mult >= 1.8:
                pos.sl = min(pos.sl, mkt.ac[i] + 0.9 * risk)


class AsianBreakout(Base):
    """S1 - break of the Asian session range during the London session.

    Gold's Asian session is quiet and its London session is directional, so the
    overnight range is a genuine decision level. Filters reject days where the
    range is so wide the move has already happened, or so narrow the break is
    noise.
    """

    name = "S1 Asian Range Breakout (London)"
    risk_pct = 0.0060
    max_trades_day = 1
    sessions = ((7, 12),)
    exit_hour = 17

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.d1 = resample(mkt, 1440)
        self.atr_h1 = atr(self.h1, 14)
        self.atr_d1 = atr(self.d1, 14)
        self.hi, self.lo = session_range(mkt, 0, 7)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        hi, lo = self.hi[i], self.lo[i]
        if not np.isfinite(hi) or not np.isfinite(lo):
            return None
        k1, kd = self.h1.idx_map[i], self.d1.idx_map[i]
        if k1 < 20 or kd < 20:
            return None
        a1, ad = self.atr_h1[k1], self.atr_d1[kd]
        if a1 <= 0 or ad <= 0:
            return None

        width = hi - lo
        if not (0.15 * ad <= width <= 0.85 * ad):
            return None

        buf = 0.12 * a1
        c = mkt.c[i]
        sl_dist = float(np.clip(max(0.40 * width, 0.65 * a1), 0.5 * a1, 1.6 * a1))
        risk = self.risk_for(ctx)
        if c > hi + buf:
            return (1, sl_dist, 2.0 * sl_dist, risk, "asia_break_up")
        if c < lo - buf:
            return (-1, sl_dist, 2.0 * sl_dist, risk, "asia_break_dn")
        return None

    def force_exit(self, pos, i, mkt) -> bool:
        return mkt.hour[i] >= self.exit_hour


class TrendPullback(Base):
    """S2 - buy pullbacks inside an established H1/H4 trend.

    Trades with the dominant trend only, entering on a resumption bar after
    price pulls back toward the H1 EMA20, so the stop sits behind structure
    rather than at an arbitrary distance.
    """

    name = "S2 H1 Trend Pullback"
    risk_pct = 0.0060
    max_trades_day = 2
    sessions = ((7, 19),)

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.h4 = resample(mkt, 240)
        self.m15 = resample(mkt, 15)
        self.ema20 = ema(self.h1.c, 20)
        self.ema50 = ema(self.h1.c, 50)
        self.ema_h4 = ema(self.h4.c, 50)
        self.atr_h1 = atr(self.h1, 14)
        self.m15_hi = rolling_max(self.m15.h, 4)
        self.m15_lo = rolling_min(self.m15.l, 4)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1, k4, k15 = self.h1.idx_map[i], self.h4.idx_map[i], self.m15.idx_map[i]
        if k1 < 60 or k4 < 55 or k15 < 10:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0:
            return None
        e20, e50, e4 = self.ema20[k1], self.ema50[k1], self.ema_h4[k4]
        c1, c4 = self.h1.c[k1], self.h4.c[k4]
        c = mkt.c[i]
        sl_dist = float(np.clip(1.1 * a1, 0.8 * a1, 2.0 * a1))
        risk = self.risk_for(ctx)

        up = e20 > e50 and c1 > e50 and c4 > e4
        dn = e20 < e50 and c1 < e50 and c4 < e4

        # must be near the mean (not extended) and breaking the recent m15 range
        if up and abs(c - e20) < 1.2 * a1 and c > self.m15_hi[k15]:
            return (1, sl_dist, 2.4 * sl_dist, risk, "trend_pb_long")
        if dn and abs(c - e20) < 1.2 * a1 and c < self.m15_lo[k15]:
            return (-1, sl_dist, 2.4 * sl_dist, risk, "trend_pb_short")
        return None


class DonchianTrend(Base):
    """S3 - H1 Donchian breakout in the direction of the daily trend.

    Few trades, long holds, designed to capture the multi-week directional legs
    that dominate gold. Weekend holding is permitted by the firm, so positions
    are allowed to run.
    """

    name = "S3 H1 Donchian Trend"
    risk_pct = 0.0055
    max_trades_day = 1
    sessions = ((1, 21),)

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.d1 = resample(mkt, 1440)
        self.atr_h1 = atr(self.h1, 14)
        self.dc_hi = rolling_max(self.h1.h, 30)
        self.dc_lo = rolling_min(self.h1.l, 30)
        self.adx_h1 = adx(self.h1, 14)
        self.ema_d1 = ema(self.d1.c, 20)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1, kd = self.h1.idx_map[i], self.d1.idx_map[i]
        if k1 < 40 or kd < 25:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0 or not np.isfinite(self.dc_hi[k1]):
            return None
        if self.adx_h1[k1] < 18:
            return None
        c = mkt.c[i]
        d_up = self.d1.c[kd] > self.ema_d1[kd]
        sl_dist = float(np.clip(1.5 * a1, 1.0 * a1, 2.5 * a1))
        risk = self.risk_for(ctx)
        if d_up and c > self.dc_hi[k1]:
            return (1, sl_dist, 3.0 * sl_dist, risk, "donch_long")
        if (not d_up) and c < self.dc_lo[k1]:
            return (-1, sl_dist, 3.0 * sl_dist, risk, "donch_short")
        return None


class RangeFade(Base):
    """S4 - fade stretched moves, but only in a confirmed range regime.

    Deliberately decorrelated from S1-S3: it only trades when ADX says there is
    no trend, and it targets the mean rather than a breakout extension.
    """

    name = "S4 Range Fade (mean reversion)"
    risk_pct = 0.0050
    max_trades_day = 2
    sessions = ((7, 18),)
    exit_hour = 20

    def prepare(self, mkt):
        self.h1 = resample(mkt, 60)
        self.m15 = resample(mkt, 15)
        self.atr_h1 = atr(self.h1, 14)
        self.adx_h1 = adx(self.h1, 14)
        self.rsi15 = rsi(self.m15.c, 14)
        self.mean15 = ema(self.m15.c, 96)  # ~24h mean on 15m bars

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k1, k15 = self.h1.idx_map[i], self.m15.idx_map[i]
        if k1 < 40 or k15 < 120:
            return None
        a1 = self.atr_h1[k1]
        if a1 <= 0:
            return None
        if self.adx_h1[k1] > 22:  # trending: stand aside
            return None
        c = mkt.c[i]
        m = self.mean15[k15]
        z = (c - m) / a1
        r = self.rsi15[k15]
        sl_dist = float(np.clip(1.2 * a1, 0.9 * a1, 2.0 * a1))
        risk = self.risk_for(ctx)
        if z > 1.8 and r > 68:
            return (-1, sl_dist, 1.5 * sl_dist, risk, "fade_short")
        if z < -1.8 and r < 32:
            return (1, sl_dist, 1.5 * sl_dist, risk, "fade_long")
        return None

    def force_exit(self, pos, i, mkt) -> bool:
        return mkt.hour[i] >= self.exit_hour


ALL = [AsianBreakout, TrendPullback, DonchianTrend, RangeFade]
