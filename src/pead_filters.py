"""Two filters on the PEAD event set: net share issuance, and low volatility.

Both are documented effects in their own right. Net share issuance (Pontiff &
Woodgate; Daniel & Titman) says companies shrinking their share count go on to
outperform those diluting. Low volatility (Ang et al.; Baker, Bradley &
Wurgler) says the least volatile stocks earn more per unit of risk than the
most.

A warning that belongs with the numbers rather than after them. PEAD was
already measured and already failed before these filters were written. Adding
conditions to a strategy that lost and keeping whichever combination wins is
how a backtest gets fooled: with enough filters something always survives. So
the whole grid is reported, not the best cell; every combination is measured on
2015-2020 and then re-measured unchanged on 2021-2026; and the count of cells
tried is printed alongside, because that count is what a surviving cell has to
beat.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = "data/pead"


def load_shares(path=os.path.join(DATA, "shares.json")):
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        raw = json.load(fh)
    out = {}
    for tk, rows in raw.items():
        if len(rows) < 8:
            continue
        s = pd.Series({pd.Timestamp(r["date"]): float(r["shares"]) for r in rows})
        out[tk] = s.sort_index()
    return out


def issuance_at(shares: pd.Series, when: pd.Timestamp, lookback_days=400):
    """Year-on-year net share issuance known before `when`.

    Only filings dated strictly before the event are visible, and the
    comparison point is the closest observation about a year earlier. Negative
    means the share count shrank -- a buyback.
    """
    past = shares[shares.index < when]
    if len(past) < 2:
        return np.nan
    now = past.iloc[-1]
    ref_date = past.index[-1] - pd.Timedelta(days=lookback_days)
    older = past[past.index <= ref_date]
    if older.empty or now <= 0:
        return np.nan
    then = older.iloc[-1]
    if then <= 0:
        return np.nan
    return float(np.log(now / then))


def attach_issuance(df, shares):
    """Net issuance for every event, as of the day before it."""
    vals = []
    for r in df.itertuples():
        s = shares.get(r.ticker)
        vals.append(np.nan if s is None
                    else issuance_at(s, pd.Timestamp(r.date)))
    out = df.copy()
    out["issuance"] = vals
    return out


def attach_volatility(df, excess, sessions, window=60):
    """Trailing realised volatility at the event, and its cross-sectional rank.

    Volatility is measured on the `window` sessions ending the day before the
    reaction, so it is known when the trade is placed. The rank is taken among
    the events reacting in the same calendar quarter, which is what "the least
    volatile quarter of the market" means at a point in time.
    """
    ex = excess.fillna(0.0)
    cols = {t: j for j, t in enumerate(ex.columns)}
    A = ex.to_numpy()
    vol = []
    for r in df.itertuples():
        j = cols.get(r.ticker)
        if j is None:
            vol.append(np.nan)
            continue
        lo = max(0, r.react_i - window)
        seg = A[lo:r.react_i, j]
        vol.append(float(np.std(seg, ddof=1)) if len(seg) > 20 else np.nan)
    out = df.copy()
    out["vol"] = vol
    dates = pd.DatetimeIndex(sessions)[out["react_i"].to_numpy()]
    out["quarter"] = pd.PeriodIndex(dates, freq="Q")
    out["vol_rank"] = out.groupby("quarter")["vol"].rank(pct=True)
    return out
