"""Post-earnings announcement drift, traded long/short.

The classic result: a stock that surprises on earnings keeps drifting in the
direction of the surprise for weeks afterwards. The surprise measure used here
is the stock's own market-adjusted reaction to the announcement rather than a
gap against analyst estimates -- the "earnings announcement return" variant,
which is documented in its own right and, usefully, needs no estimate data.

Three details decide whether this is a real backtest:

  * When the news becomes tradeable. An 8-K accepted at 16:30 ET is not
    information for that day's close. The reaction window starts on the next
    session whenever acceptance is after the close, which is most of the time.

  * How events are ranked. Ranking within an earnings season needs the season
    to finish, and entering before it does would be a look-ahead. Each event is
    instead scored against the distribution of announcement reactions that had
    already completed when it happened, which is strictly causal and needs no
    waiting.

  * Market adjustment. A quarter where everything rose is not evidence of drift,
    so every return is measured net of SPY on the same day.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = "data/pead"
NY = "America/New_York"

# Round-trip cost in basis points by average dollar volume. PEAD is strongest
# in exactly the illiquid names where this is largest, so it is a tier, not a
# constant.
COST_TIERS = [(1e8, 8.0), (2e7, 20.0), (5e6, 45.0), (0.0, 90.0)]


def cost_bp(adv: float) -> float:
    for floor, bp in COST_TIERS:
        if adv >= floor:
            return bp
    return COST_TIERS[-1][1]


def load_prices(tickers, path=os.path.join(DATA, "px")):
    px, adv = {}, {}
    for tk in tickers:
        f = os.path.join(path, f"{tk}.parquet")
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        idx = pd.DatetimeIndex(d["date"]).tz_localize(None)
        s = pd.Series(d["adj_close"].values, index=idx).sort_index()
        s = s[~s.index.duplicated(keep="last")]
        if s.notna().sum() < 500:
            continue
        px[tk] = s
        dv = (pd.Series(d["close"].values, index=idx)
              * pd.Series(d["volume"].values, index=idx))
        adv[tk] = float(dv.tail(750).median()) if dv.notna().any() else 0.0
    return pd.DataFrame(px).sort_index(), adv


def market_series(path="data/yahoo/SPY.parquet"):
    d = pd.read_parquet(path)
    idx = pd.DatetimeIndex(d["date"]).tz_localize(None)
    return pd.Series(d["adj_close"].values, index=idx).sort_index().pct_change()


def reaction_day(ev, sessions: pd.DatetimeIndex):
    """First session on which the announcement is tradeable.

    Acceptance timestamps are UTC. 16:00 ET is 20:00 UTC in summer and 21:00 in
    winter; using 20:00 as the cut is the conservative choice, since treating a
    borderline release as next-day can only delay entry, never advance it.
    """
    d = pd.Timestamp(ev["date"])
    acc = ev.get("accepted") or ""
    after_close = True
    if acc:
        try:
            t = pd.Timestamp(acc).tz_convert("UTC")
            after_close = t.hour >= 20
        except Exception:
            after_close = True
    # "right" lands strictly past d, "left" lands on d when d is a session and
    # on the next one otherwise -- which is what a pre-open release on a
    # holiday should do.
    pos = sessions.searchsorted(d, side="right" if after_close else "left")
    if pos >= len(sessions):
        return None
    return int(pos)


def build_events(events_json=os.path.join(DATA, "events.json"),
                 window=2, hold=60):
    """Every announcement as a row: when it became tradeable and how it reacted."""
    with open(events_json) as fh:
        ev = json.load(fh)
    tickers = sorted(ev)
    px, adv = load_prices(tickers)
    if px.empty:
        return None, None, None
    mkt = market_series().reindex(px.index).fillna(0.0)
    rets = px.pct_change()
    excess = rets.sub(mkt, axis=0)
    sessions = px.index

    rows = []
    for tk in px.columns:
        for e in ev[tk]["events"]:
            i = reaction_day(e, sessions)
            if i is None or i + window + hold >= len(sessions):
                continue
            seg = excess[tk].iloc[i:i + window]
            if seg.isna().any():
                continue
            rows.append({
                "ticker": tk, "date": e["date"],
                "react_i": i, "entry_i": i + window,
                "reaction": float((1 + seg).prod() - 1),
                "adv": adv.get(tk, 0.0),
            })
    df = pd.DataFrame(rows).sort_values("react_i").reset_index(drop=True)
    return df, excess, sessions


def causal_percentile(df, min_history=200):
    """Score each event against reactions that had already finished before it.

    Expanding, not rolling on the whole sample: at event k only events whose
    entry index is at or before event k's reaction index are visible.
    """
    order = df.sort_values("react_i").index
    reactions = df.loc[order, "reaction"].to_numpy()
    entry = df.loc[order, "entry_i"].to_numpy()
    react = df.loc[order, "react_i"].to_numpy()

    pct = np.full(len(order), np.nan)
    for k in range(len(order)):
        # events already complete when event k reacts
        done = entry[:k] <= react[k]
        if done.sum() < min_history:
            continue
        prior = reactions[:k][done]
        pct[k] = (prior < reactions[k]).mean()
    out = pd.Series(pct, index=order).sort_index()
    return out


def run(top=0.2, hold=60, window=2, apply_costs=True, min_adv=0.0):
    df, excess, sessions = build_events(window=window, hold=hold)
    if df is None or df.empty:
        return None
    if min_adv > 0:
        df = df[df["adv"] >= min_adv].reset_index(drop=True)
    df["pct"] = causal_percentile(df)
    df = df.dropna(subset=["pct"]).reset_index(drop=True)
    df["side"] = np.where(df["pct"] >= 1 - top, 1.0,
                          np.where(df["pct"] <= top, -1.0, 0.0))
    trades = df[df["side"] != 0].reset_index(drop=True)

    n = len(sessions)
    cols = sorted(trades["ticker"].unique())
    col_of = {t: j for j, t in enumerate(cols)}
    # Accumulate into a plain array: thousands of overlapping holding windows
    # written through DataFrame.loc is minutes of work for the same result.
    arr = np.zeros((n, len(cols)))
    for r in trades.itertuples():
        lo, hi = r.entry_i, min(r.entry_i + hold, n)
        arr[lo:hi, col_of[r.ticker]] += r.side
    pos = pd.DataFrame(arr, columns=cols)

    active = (pos != 0)
    count = active.sum(axis=1).replace(0, np.nan)
    w = pos.div(count, axis=0).fillna(0.0)
    w.index = sessions

    ex = excess.reindex(columns=w.columns).fillna(0.0)
    gross = (w.shift(1).fillna(0.0) * ex).sum(axis=1)

    if apply_costs:
        bp = pd.Series({t: cost_bp(a) for t, a in
                        zip(trades["ticker"], trades["adv"])})
        bp = bp.reindex(w.columns).fillna(COST_TIERS[-1][1])
        turn = w.diff().abs().fillna(w.abs())
        fees = (turn * bp / 1e4).sum(axis=1)
    else:
        fees = pd.Series(0.0, index=w.index)

    return {"ret": (gross - fees).rename("pead"), "gross": gross,
            "fees": fees, "trades": trades, "events": df,
            "n_positions": active.sum(axis=1)}


if __name__ == "__main__":
    from multistrat import fmt_stats, stats
    res = run()
    if res is None:
        raise SystemExit("no data yet -- run src/pead_data.py first")
    tr = res["trades"]
    print(f"{len(res['events']):,} scored announcements across "
          f"{res['events']['ticker'].nunique()} names")
    print(f"{len(tr):,} traded ({int((tr['side']>0).sum())} long, "
          f"{int((tr['side']<0).sum())} short)")
    print(f"median positions open: {res['n_positions'].median():.0f}, "
          f"max {res['n_positions'].max():.0f}\n")
    print(fmt_stats([("PEAD gross", stats(res["gross"])),
                     ("PEAD after costs", stats(res["ret"]))]))
