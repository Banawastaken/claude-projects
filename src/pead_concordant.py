"""Post-earnings drift under the published rules, including the concordance filter.

The rules as stated:

  long  when EPS beats consensus AND the stock reacts up on the reaction day
  short when EPS misses consensus AND the stock reacts down
  nothing at all when the two disagree

  reaction day is the announcement day for a pre-market release and the next
  session for a post-market one; entry is the following session's open
  hold exactly 60 trading days, no stop and no target
  flat 10% of capital per position

The concordance filter is the part nothing free could reach before: it needs a
consensus estimate per quarter, which Alpha Vantage supplies along with the
pre/post-market flag that decides which session reacts.

Whether returns are market-adjusted is not specified anywhere, and it is worth
more than any parameter here -- over this sample the market contributes roughly
13 points a year to a long book -- so both are always reported.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pead import load_prices, market_series  # noqa: E402

DATA = "data/pead"


def build(events_path=os.path.join(DATA, "av_earnings.json"),
          hold=60, market_adjust=True):
    """One row per announcement: its surprise, its reaction, and its side."""
    with open(events_path) as fh:
        ev = json.load(fh)
    px, adv, _ = load_prices(sorted(ev))
    if px.empty:
        return None, None, None
    rets = px.pct_change()
    if market_adjust:
        mkt = market_series().reindex(px.index).fillna(0.0)
        rets = rets.sub(mkt, axis=0)
    sessions = px.index

    rows = []
    for tk in px.columns:
        for e in ev.get(tk, []):
            d = pd.Timestamp(e["date"])
            # Pre-market: the announcement day itself reacts. Post-market: the
            # next session does, because the news lands after the close.
            side = "left" if e["before_open"] else "right"
            i = sessions.searchsorted(d, side=side)
            if i >= len(sessions) or i + 1 + hold >= len(sessions):
                continue
            reaction = rets[tk].iloc[i]
            if not np.isfinite(reaction):
                continue
            rows.append({
                "ticker": tk, "date": e["date"],
                "before_open": e["before_open"],
                "surprise": e["surprise"],
                "surprise_pct": e["surprise_pct"],
                "reaction": float(reaction),
                "react_i": int(i), "entry_i": int(i) + 1,
            })
    df = pd.DataFrame(rows).sort_values("entry_i").reset_index(drop=True)
    return df, rets, sessions


def sides(df, concordant=True, min_surprise=0.0):
    """Trade direction under the concordance rule, or price reaction alone."""
    beat = df["surprise"] > min_surprise
    miss = df["surprise"] < -min_surprise
    up = df["reaction"] > 0
    down = df["reaction"] < 0
    if concordant:
        return np.where(beat & up, 1.0, np.where(miss & down, -1.0, 0.0))
    return np.where(up, 1.0, np.where(down, -1.0, 0.0))


def backtest(df, rets, sessions, side, hold=60, flat_weight=0.10,
             cost_bp=8.0):
    """Flat weight per position, held 60 sessions, no stop."""
    d = df.copy()
    d["side"] = side
    tr = d[d["side"] != 0].reset_index(drop=True)
    if tr.empty:
        return None

    n = len(sessions)
    cols = sorted(tr["ticker"].unique())
    col_of = {t: j for j, t in enumerate(cols)}
    arr = np.zeros((n, len(cols)))
    for r in tr.itertuples():
        lo, hi = r.entry_i, min(r.entry_i + hold, n)
        arr[lo:hi, col_of[r.ticker]] += r.side
    pos = pd.DataFrame(arr, index=sessions, columns=cols)
    w = pos * flat_weight

    ex = rets.reindex(columns=cols).fillna(0.0)
    gross = (w.shift(1).fillna(0.0) * ex).sum(axis=1)
    turn = pos.diff().abs().fillna(pos.abs()) * flat_weight
    fees = (turn * cost_bp / 1e4).sum(axis=1)
    return {"ret": (gross - fees).rename("pead"), "gross": gross,
            "trades": tr, "exposure": w.abs().sum(axis=1)}
