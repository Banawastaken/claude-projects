"""Vectorised indicators over M1 bars, with strict no-lookahead resampling.

`resample` returns higher-timeframe OHLC arrays plus a map from every M1 index
to the last *completed* higher-timeframe bar. Reading indicators through that
map guarantees a strategy never sees a bar that is still forming.
"""

from __future__ import annotations

import numpy as np


class TF:
    """A higher timeframe view of the M1 series."""

    def __init__(self, o, h, l, c, idx_map, start_idx):
        self.o, self.h, self.l, self.c = o, h, l, c
        self.idx_map = idx_map  # M1 index -> index of last COMPLETED HTF bar
        self.start_idx = start_idx  # M1 index where each HTF bar began
        self.n = len(c)


def resample(mkt, minutes: int) -> TF:
    epoch_min = mkt.ts.astype("datetime64[m]").astype(np.int64)
    bucket = epoch_min // minutes
    # boundaries where the bucket changes
    change = np.empty(len(bucket), dtype=bool)
    change[0] = True
    change[1:] = bucket[1:] != bucket[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(bucket))

    n = len(starts)
    o = mkt.o[starts]
    c = mkt.c[ends - 1]
    h = np.maximum.reduceat(mkt.h, starts)
    l = np.minimum.reduceat(mkt.l, starts)

    # map every M1 bar to the index of the last completed HTF bar (current - 1)
    bar_of_m1 = np.repeat(np.arange(n), ends - starts)
    idx_map = bar_of_m1 - 1
    return TF(o, h, l, c, idx_map, starts)


def ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def rma(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing."""
    alpha = 1.0 / period
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def atr(tf: TF, period: int = 14) -> np.ndarray:
    prev_c = np.concatenate(([tf.c[0]], tf.c[:-1]))
    tr = np.maximum.reduce(
        [tf.h - tf.l, np.abs(tf.h - prev_c), np.abs(tf.l - prev_c)]
    )
    return rma(tr, period)


def rsi(x: np.ndarray, period: int = 14) -> np.ndarray:
    d = np.diff(x, prepend=x[0])
    gain = rma(np.clip(d, 0, None), period)
    loss = rma(np.clip(-d, 0, None), period)
    rs = gain / np.where(loss == 0, 1e-12, loss)
    return 100.0 - 100.0 / (1.0 + rs)


def adx(tf: TF, period: int = 14) -> np.ndarray:
    up = np.diff(tf.h, prepend=tf.h[0])
    dn = -np.diff(tf.l, prepend=tf.l[0])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    prev_c = np.concatenate(([tf.c[0]], tf.c[:-1]))
    tr = np.maximum.reduce([tf.h - tf.l, np.abs(tf.h - prev_c), np.abs(tf.l - prev_c)])
    atr_ = rma(tr, period)
    atr_safe = np.where(atr_ == 0, 1e-12, atr_)
    pdi = 100.0 * rma(plus_dm, period) / atr_safe
    mdi = 100.0 * rma(minus_dm, period) / atr_safe
    dx = 100.0 * np.abs(pdi - mdi) / np.where((pdi + mdi) == 0, 1e-12, pdi + mdi)
    return rma(dx, period)


def rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Max of the previous `window` values, excluding the current one."""
    out = np.full_like(x, np.nan)
    if len(x) <= window:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    sw = sliding_window_view(x[:-1], window)
    out[window:] = sw.max(axis=1)
    return out


def rolling_min(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if len(x) <= window:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    sw = sliding_window_view(x[:-1], window)
    out[window:] = sw.min(axis=1)
    return out


def session_range(mkt, start_hour: int, end_hour: int):
    """Per-day high/low of the [start_hour, end_hour) UTC window.

    Returns arrays indexed by M1 bar holding the range of the CURRENT day's
    session, valid only at bars at or after `end_hour` (NaN before that, so a
    strategy cannot read a session that has not finished).
    """
    day = mkt.ts.astype("datetime64[D]")
    hour = mkt.hour
    in_sess = (hour >= start_hour) & (hour < end_hour)

    hi = np.full(len(day), np.nan)
    lo = np.full(len(day), np.nan)

    # unique day boundaries
    change = np.empty(len(day), dtype=bool)
    change[0] = True
    change[1:] = day[1:] != day[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(day))

    for s, e in zip(starts, ends):
        m = in_sess[s:e]
        if not m.any():
            continue
        h = mkt.h[s:e][m].max()
        l = mkt.l[s:e][m].min()
        after = np.flatnonzero(hour[s:e] >= end_hour)
        if len(after) == 0:
            continue
        hi[s + after[0] : e] = h
        lo[s + after[0] : e] = l
    return hi, lo
