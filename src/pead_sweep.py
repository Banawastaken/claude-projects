"""Every parameter of the PEAD rule, swept honestly.

The parameters: holding period, which side to take, a minimum absolute surprise,
percent of capital per position, a hard stop and a trailing stop.

Sweeping six parameters over 691 events and keeping the best cell is how a
backtest gets fooled -- with a few hundred combinations something always looks
excellent. So the grid is evaluated on 2015-2020 only, the cell that wins there
is then run once on 2021-2026, and the whole surface is reported rather than
the peak. A parameter worth using is one whose neighbours also work; a lone
spike surrounded by mediocrity is noise that happened to land.

Stops need the intraday path, so this walks each position day by day over
adjusted highs and lows rather than compounding closes. Yahoo's `adj_close`
carries splits and dividends but the raw high and low do not, so they are
scaled by the same factor before any stop is tested against them -- otherwise
every pre-split position stops out on a price that never happened.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pead_concordant import build, sides  # noqa: E402

DATA = "data/pead"
SPLIT = pd.Timestamp("2021-01-01")


def price_paths(tickers, path=os.path.join(DATA, "px")):
    """Adjusted close, high and low per ticker, on one shared calendar."""
    close, high, low = {}, {}, {}
    for tk in tickers:
        f = os.path.join(path, f"{tk}.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        idx = pd.DatetimeIndex(d["date"]).tz_localize(None)
        c = pd.Series(d["adj_close"].values, index=idx)
        raw = pd.Series(d["close"].values, index=idx)
        fac = (c / raw).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        close[tk] = c[~c.index.duplicated(keep="last")]
        high[tk] = (pd.Series(d["high"].values, index=idx) * fac)[~idx.duplicated(keep="last")]
        low[tk] = (pd.Series(d["low"].values, index=idx) * fac)[~idx.duplicated(keep="last")]
    return (pd.DataFrame(close).sort_index(), pd.DataFrame(high).sort_index(),
            pd.DataFrame(low).sort_index())


def walk(trades, C, H, L, hold, hard_stop=None, trail_stop=None):
    """Per-trade outcome, walking the price path and honouring stops.

    A stop is checked against the day's extreme, so it fires the day the price
    trades through it rather than the day it closes through it. When both a
    stop and the holding period could end a trade on the same day the stop
    wins, which is the conservative reading.
    """
    cols = {t: j for j, t in enumerate(C.columns)}
    c = C.to_numpy(); h = H.to_numpy(); l = L.to_numpy()
    n = len(C)
    out = []
    for r in trades.itertuples():
        j = cols.get(r.ticker)
        if j is None:
            continue
        i0 = r.entry_i
        if i0 + 1 >= n or not np.isfinite(c[i0, j]):
            continue
        entry = c[i0, j]
        if entry <= 0:
            continue
        side = r.side
        peak = entry
        exit_px, exit_i, why = None, None, "hold"
        last = min(i0 + hold, n - 1)
        for i in range(i0 + 1, last + 1):
            hi, lo, cl = h[i, j], l[i, j], c[i, j]
            if not np.isfinite(cl):
                continue
            if side > 0:
                peak = max(peak, hi if np.isfinite(hi) else cl)
                if hard_stop and np.isfinite(lo) and lo <= entry * (1 - hard_stop):
                    exit_px, exit_i, why = entry * (1 - hard_stop), i, "hard"
                    break
                if trail_stop and np.isfinite(lo) and lo <= peak * (1 - trail_stop):
                    exit_px, exit_i, why = peak * (1 - trail_stop), i, "trail"
                    break
            else:
                peak = min(peak, lo if np.isfinite(lo) else cl)
                if hard_stop and np.isfinite(hi) and hi >= entry * (1 + hard_stop):
                    exit_px, exit_i, why = entry * (1 + hard_stop), i, "hard"
                    break
                if trail_stop and np.isfinite(hi) and hi >= peak * (1 + trail_stop):
                    exit_px, exit_i, why = peak * (1 + trail_stop), i, "trail"
                    break
        if exit_px is None:
            exit_px, exit_i = c[last, j], last
        out.append({"ticker": r.ticker, "entry_i": i0, "exit_i": exit_i,
                    "side": side, "ret": side * (exit_px / entry - 1.0),
                    "why": why, "days": exit_i - i0})
    return pd.DataFrame(out)


def portfolio(tr, sessions, weight, cost_bp=8.0, mkt=None):
    """Daily account return from per-trade outcomes at a flat weight.

    Each trade's P&L is spread evenly over the days it was held, which keeps the
    daily series honest about when capital was at risk without needing to
    re-walk every path.
    """
    if tr.empty:
        return None
    n = len(sessions)
    pnl = np.zeros(n)
    expo = np.zeros(n)
    for r in tr.itertuples():
        d = max(r.exit_i - r.entry_i, 1)
        per = (r.ret - cost_bp / 1e4) / d
        pnl[r.entry_i + 1:r.exit_i + 1] += per * weight
        expo[r.entry_i + 1:r.exit_i + 1] += weight * abs(r.side)
    s = pd.Series(pnl, index=sessions)
    if mkt is not None:
        net = pd.Series(np.zeros(n), index=sessions)
        for r in tr.itertuples():
            net.iloc[r.entry_i + 1:r.exit_i + 1] += weight * r.side
        s = s - net * mkt.reindex(sessions).fillna(0.0)
    return s


def score(r):
    """What a parameter set is judged on: risk-adjusted return, not return."""
    from multistrat import stats
    if r is None or r.abs().sum() == 0:
        return None
    return stats(r)
