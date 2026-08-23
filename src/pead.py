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

# A sample drawn at random from every EDGAR filer is mostly micro-caps, and
# Yahoo's adjusted close does not reliably handle their reverse splits: the raw
# pull contained a +12,213% day. A move that size is a corporate-action
# artifact rather than a return, and the series carrying it cannot be trusted
# anywhere, so the name is dropped whole rather than the day patched. The
# threshold is set far above any genuine one-day move so it catches errors and
# not biotech.
MAX_PLAUSIBLE_DAILY = 5.0

# Tradeability floor. PEAD is strongest in illiquid names, so this knowingly
# gives up measured effect for a number that could be realised.
DEFAULT_MIN_ADV = 5e6


def cost_bp(adv: float) -> float:
    for floor, bp in COST_TIERS:
        if adv >= floor:
            return bp
    return COST_TIERS[-1][1]


def load_prices(tickers, path=os.path.join(DATA, "px")):
    px, adv, dropped = {}, {}, []
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
        r = s.pct_change()
        if r.abs().max() > MAX_PLAUSIBLE_DAILY:
            dropped.append((tk, float(r.abs().max())))
            continue
        px[tk] = s
        dv = (pd.Series(d["close"].values, index=idx)
              * pd.Series(d["volume"].values, index=idx))
        adv[tk] = float(dv.tail(750).median()) if dv.notna().any() else 0.0
    return pd.DataFrame(px).sort_index(), adv, dropped


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
                 window=2, hold=60, market_adjust=True):
    """Every announcement as a row: when it became tradeable and how it reacted."""
    with open(events_json) as fh:
        ev = json.load(fh)
    tickers = sorted(ev)
    px, adv, dropped = load_prices(tickers)
    if px.empty:
        return None, None, None
    if dropped:
        print(f"  dropped {len(dropped)} names with implausible daily moves "
              f"(worst {max(d for _, d in dropped)*100:,.0f}%)")
    rets = px.pct_change()
    if market_adjust:
        mkt = market_series().reindex(px.index).fillna(0.0)
        excess = rets.sub(mkt, axis=0)
    else:
        # Raw returns carry the market. Over 2015-2026 that is roughly +14% a
        # year of beta arriving in the long leg, which a long-only book keeps
        # and a dollar-neutral one cancels.
        excess = rets
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


def run(top=0.2, hold=60, window=2, apply_costs=True,
        min_adv=DEFAULT_MIN_ADV, max_adv=None,
        issuance_mode=None, vol_cut=None,
        market_adjust=True, long_only=False, flat_weight=None):
    df, excess, sessions = build_events(window=window, hold=hold,
                                        market_adjust=market_adjust)
    if df is None or df.empty:
        return None
    before = df["ticker"].nunique()
    if min_adv > 0:
        df = df[df["adv"] >= min_adv]
    if max_adv is not None:
        df = df[df["adv"] < max_adv]
    df = df.reset_index(drop=True)
    if min_adv > 0 or max_adv is not None:
        hi = "inf" if max_adv is None else f"{max_adv/1e6:.0f}M"
        print(f"  liquidity band ${min_adv/1e6:.0f}M-${hi} ADV: "
              f"{df['ticker'].nunique()} of {before} names kept")
    if issuance_mode or vol_cut is not None:
        from pead_filters import attach_issuance, attach_volatility, load_shares
        df = attach_volatility(attach_issuance(df, load_shares()),
                               excess, sessions)
        if vol_cut is not None:
            df = df[df["vol_rank"] <= vol_cut]
        if issuance_mode:
            df = df[df["issuance"].notna()]
        df = df.reset_index(drop=True)

    df["pct"] = causal_percentile(df)
    df = df.dropna(subset=["pct"]).reset_index(drop=True)
    df["side"] = np.where(df["pct"] >= 1 - top, 1.0,
                          np.where(df["pct"] <= top, -1.0, 0.0))
    if long_only:
        df.loc[df["side"] < 0, "side"] = 0.0
    if issuance_mode == "aligned":
        # Only take the events where the earnings reaction and the share count
        # point the same way: a good surprise from a company buying back, a bad
        # one from a company diluting.
        keep = ((df["side"] > 0) & (df["issuance"] < 0)) | \
               ((df["side"] < 0) & (df["issuance"] > 0))
        df.loc[~keep, "side"] = 0.0
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
    # Carry the session dates from here on, so every frame derived from `pos`
    # shares one index. Setting it later left `count` on the integer index and
    # silently reindexed the fee calculation to all-NaN, billing nothing.
    pos = pd.DataFrame(arr, index=sessions, columns=cols)

    active = (pos != 0)
    count = active.sum(axis=1).replace(0, np.nan)
    if flat_weight:
        # A fixed fraction of capital per position, as the published rules
        # specify. Gross exposure is then however many positions happen to be
        # open times that fraction, so the book levers up in busy earnings
        # season rather than diluting each name.
        w = (pos * flat_weight).fillna(0.0)
    else:
        w = pos.div(count, axis=0).fillna(0.0)

    ex = excess.reindex(columns=w.columns).fillna(0.0)
    gross = (w.shift(1).fillna(0.0) * ex).sum(axis=1)

    if apply_costs:
        bp = pd.Series({t: cost_bp(a) for t, a in
                        zip(trades["ticker"], trades["adv"])})
        bp = bp.reindex(w.columns).fillna(COST_TIERS[-1][1])
        # Charge only real entries and exits. Taking turnover off the
        # normalised weights instead would re-mark every open position on any
        # day the position count changed -- which is most days -- and bill a
        # rebalance nobody would place.
        opened = pos.diff().abs().fillna(pos.abs())
        turn = opened.div(count, axis=0).fillna(0.0)
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
