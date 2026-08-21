"""Candidate concepts for the second attempt.

Chosen for a reason rather than invented: the first attempt searched chart
patterns and found one that fitted its sample. These are effects with prior
evidence outside this dataset -- documented across decades and asset classes by
people who were not looking at 2025 gold.

* `TSMomentum` -- time-series momentum. The most replicated cross-asset anomaly
  there is (Moskowitz, Ooi & Pedersen), tested over a century of futures data.
* `TurnOfMonth` -- the turn-of-the-month effect in equity indices, attributed to
  pension and payroll flows, documented since the 1980s and structural rather
  than technical.
* `ShortTermReversal` -- buying weakness inside an established uptrend. Well
  documented in index products, and the opposite sign to the breakout logic
  that failed, so it fails differently if it fails.
* `VolContraction` -- expansion after compression. A volatility effect rather
  than a directional one, so its edge does not depend on predicting direction.

The exploratory scan found no durable autocorrelation in gold at any horizon
from one hour to one week, and confirmed the apparent 22:00 UTC session edge is
a spread artifact at broker rollover. Both are reasons not to expect much from
pure price-pattern rules on gold, and reasons to test breadth across many
instruments instead.
"""

from __future__ import annotations

import numpy as np

from indicators import atr, ema, resample, rolling_max, rolling_min
from strategies import Base
from strategies_v2 import TrendExitMixin


class TSMomentum(TrendExitMixin, Base):
    """Time-series momentum: hold the direction of the trailing return.

    Entry is a resumption bar in the direction of the long-horizon return, so
    the strategy is not buying the very top of an extended move.
    """

    name = "TSMomentum"
    risk_pct = 0.0075
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((1, 20),)   # 20-21 UTC excluded: rollover spreads

    lookback_days = 60      # trailing return horizon, in daily bars
    entry_bars = 6          # break of this many H4 bars to trigger
    atr_mult = 2.0
    tp_r = 4.0
    trail_start_r = 2.0
    trail_dist_r = 1.5

    def prepare(self, mkt):
        self.h4 = resample(mkt, 240)
        self.d1 = resample(mkt, 1440)
        self.atr_h4 = atr(self.h4, 14)
        self.hh = rolling_max(self.h4.h, self.entry_bars)
        self.ll = rolling_min(self.h4.l, self.entry_bars)
        # trailing return over the lookback, on completed daily bars
        c = self.d1.c
        n = len(c)
        past = np.full(n, np.nan)
        if n > self.lookback_days:
            past[self.lookback_days:] = c[: n - self.lookback_days]
        self.mom = np.where(np.isfinite(past) & (past > 0), c / past - 1.0, np.nan)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k4, kd = self.h4.idx_map[i], self.d1.idx_map[i]
        if k4 < self.entry_bars + 20 or kd < self.lookback_days + 5:
            return None
        a4 = self.atr_h4[k4]
        if a4 <= 0 or not np.isfinite(self.mom[kd]) or not np.isfinite(self.hh[k4]):
            return None
        c = mkt.c[i]
        sl_dist = float(np.clip(self.atr_mult * a4, 0.5 * a4, 4.0 * a4))
        risk = self.risk_for(ctx)
        if self.mom[kd] > 0 and c > self.hh[k4]:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "tsmom_long")
        if self.mom[kd] < 0 and c < self.ll[k4]:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "tsmom_short")
        return None


class TurnOfMonth(Base):
    """Long into the turn of the month, flat the rest of the time.

    Buys near the close of the last few trading days of a month and exits a few
    days into the next. Attributed to pension contributions and payroll flows,
    so it is a calendar rule with a mechanism rather than a fitted pattern.
    """

    name = "TurnOfMonth"
    risk_pct = 0.0075
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((1, 20),)

    days_before = 2         # start this many trading days before month end
    days_after = 3          # hold this many trading days into the new month
    atr_mult = 2.0
    tp_r = 3.0
    entry_hour = 15         # enter late in the session

    def prepare(self, mkt):
        self.d1 = resample(mkt, 1440)
        self.atr_d1 = atr(self.d1, 14)
        # position of each daily bar relative to its month boundary
        ts = mkt.ts[self.d1.start_idx].astype("datetime64[D]")
        months = ts.astype("datetime64[M]")
        n = len(months)
        self.days_to_end = np.zeros(n, dtype=int)
        self.days_from_start = np.zeros(n, dtype=int)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and months[j + 1] == months[i]:
                j += 1
            for k in range(i, j + 1):
                self.days_to_end[k] = j - k
                self.days_from_start[k] = k - i
            i = j + 1
        self._held_from = None

    def reset(self):
        super().reset()
        self._entry_day = None

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        if mkt.hour[i] != self.entry_hour:
            return None
        kd = self.d1.idx_map[i]
        if kd < 30:
            return None
        ad = self.atr_d1[kd]
        if ad <= 0:
            return None
        # the window opens `days_before` trading days from month end
        if self.days_to_end[kd] > self.days_before:
            return None
        if self.days_to_end[kd] < 0:
            return None
        sl_dist = float(np.clip(self.atr_mult * ad, 0.5 * ad, 4.0 * ad))
        self._entry_day = kd
        return (1, sl_dist, self.tp_r * sl_dist, self.risk_for(ctx), "tom_long")

    def manage(self, pos, i, mkt, rules):
        pass  # held to the time exit

    def force_exit(self, pos, i, mkt) -> bool:
        kd = self.d1.idx_map[i]
        if kd < 0:
            return False
        # out once we are far enough into the new month
        return self.days_from_start[kd] >= self.days_after and \
            self.days_to_end[kd] > self.days_before


class ShortTermReversal(Base):
    """Buy weakness inside an established uptrend, sell strength in a downtrend.

    The opposite sign to a breakout, so if the breakout family fails because
    gold whipsaws, this should benefit from the same behaviour.
    """

    name = "ShortTermReversal"
    risk_pct = 0.0075
    max_trades_day = 1
    max_losses_day = 2
    sessions = ((1, 20),)

    trend_len = 100         # daily EMA defining the regime
    pull_bars = 3           # consecutive H4 bars against the trend
    atr_mult = 2.0
    tp_r = 2.0

    def prepare(self, mkt):
        self.h4 = resample(mkt, 240)
        self.d1 = resample(mkt, 1440)
        self.atr_h4 = atr(self.h4, 14)
        self.ema_d = ema(self.d1.c, self.trend_len)
        c = self.h4.c
        o = self.h4.o
        down = c < o
        up = c > o
        n = len(c)
        self.run_down = np.zeros(n, dtype=int)
        self.run_up = np.zeros(n, dtype=int)
        for k in range(1, n):
            self.run_down[k] = self.run_down[k - 1] + 1 if down[k] else 0
            self.run_up[k] = self.run_up[k - 1] + 1 if up[k] else 0

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k4, kd = self.h4.idx_map[i], self.d1.idx_map[i]
        if k4 < 20 or kd < self.trend_len + 5:
            return None
        a4 = self.atr_h4[k4]
        if a4 <= 0:
            return None
        up_regime = self.d1.c[kd] > self.ema_d[kd]
        sl_dist = float(np.clip(self.atr_mult * a4, 0.5 * a4, 4.0 * a4))
        risk = self.risk_for(ctx)
        if up_regime and self.run_down[k4] >= self.pull_bars:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "str_long")
        if (not up_regime) and self.run_up[k4] >= self.pull_bars:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "str_short")
        return None


class VolContraction(TrendExitMixin, Base):
    """Trade the expansion that follows a volatility squeeze.

    Conditions on volatility rather than direction: when recent range collapses
    relative to its own history, take the break either way.
    """

    name = "VolContraction"
    risk_pct = 0.0075
    max_trades_day = 1
    max_losses_day = 1
    sessions = ((1, 20),)

    squeeze_len = 6         # H4 bars forming the coil
    ref_len = 60            # compare against this much history
    squeeze_ratio = 0.6     # coil range below this fraction of normal
    atr_mult = 1.5
    tp_r = 3.0
    trail_start_r = 2.0
    trail_dist_r = 1.3

    def prepare(self, mkt):
        self.h4 = resample(mkt, 240)
        self.atr_h4 = atr(self.h4, 14)
        self.hh = rolling_max(self.h4.h, self.squeeze_len)
        self.ll = rolling_min(self.h4.l, self.squeeze_len)
        width = self.hh - self.ll
        n = len(width)
        ref = np.full(n, np.nan)
        for k in range(self.ref_len, n):
            w = width[k - self.ref_len:k]
            w = w[np.isfinite(w)]
            if len(w) > 10:
                ref[k] = np.median(w)
        self.ratio = np.where(np.isfinite(ref) & (ref > 0), width / ref, np.nan)

    def signal(self, i, mkt, rules, ctx):
        if not self.allowed(i, mkt, rules, ctx):
            return None
        k4 = self.h4.idx_map[i]
        if k4 < self.ref_len + 10:
            return None
        a4 = self.atr_h4[k4]
        if a4 <= 0 or not np.isfinite(self.ratio[k4]) or not np.isfinite(self.hh[k4]):
            return None
        if self.ratio[k4] > self.squeeze_ratio:
            return None
        c = mkt.c[i]
        sl_dist = float(np.clip(self.atr_mult * a4, 0.5 * a4, 4.0 * a4))
        risk = self.risk_for(ctx)
        if c > self.hh[k4]:
            return (1, sl_dist, self.tp_r * sl_dist, risk, "squeeze_long")
        if c < self.ll[k4]:
            return (-1, sl_dist, self.tp_r * sl_dist, risk, "squeeze_short")
        return None


CANDIDATES = [TSMomentum, TurnOfMonth, ShortTermReversal, VolContraction]
