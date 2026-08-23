"""Matteo Conti's block-print framework, tested as described.

The claim: the largest trade in each minute carries information, the edge scales
with size up to a point and then breaks down, the market takes 15-20 minutes to
digest it, and the signal only works during regular hours, at the extreme of the
candle, and into a balanced book.

Everything here is measured on the print's own terms -- one observation per
minute, the largest trade in it -- rather than on a return series, because that
is how the claim is stated.

Two implementation notes that matter for reading the result:

  * the aggressor side is taken from the trade record and verified against
    contemporaneous price change rather than assumed, since getting it backwards
    would invert every number.
  * book balance comes from the matching top-of-book file, joined on the second
    the print landed. Sub-second alignment would be better and is not available
    from a one-second aggregate.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRICE_SCALE = 1e9
MIN_NS = 60_000_000_000


def trades_from_mbo(path, progress_every=10_000_000):
    """Every trade event: timestamp, price, size, aggressor side."""
    import databento as db
    store = db.DBNStore.from_file(path)
    ts, px, sz, sd = [], [], [], []
    n = 0
    for r in store:
        n += 1
        if r.action != "T":
            continue
        ts.append(r.ts_event)
        px.append(r.price)
        sz.append(r.size)
        sd.append(1 if r.side == "B" else (-1 if r.side == "A" else 0))
        if progress_every and n % progress_every == 0:
            print(f"  {n:,} records, {len(ts):,} trades", flush=True)
    return pd.DataFrame({"ts": ts, "price": np.array(px) / PRICE_SCALE,
                         "size": sz, "side": sd})


def minute_table(tr: pd.DataFrame):
    """One row per minute: the bar, and the largest print inside it."""
    tr = tr[tr["size"] > 0].copy()
    tr["minute"] = tr["ts"] // MIN_NS
    g = tr.groupby("minute")
    bar = pd.DataFrame({
        "open": g["price"].first(), "high": g["price"].max(),
        "low": g["price"].min(), "close": g["price"].last(),
        "volume": g["size"].sum(), "trades": g["size"].count(),
    })
    top = tr.loc[g["size"].idxmax()].set_index("minute")
    bar["print_size"] = top["size"]
    bar["print_side"] = top["side"]
    bar["print_price"] = top["price"]
    bar.index = pd.to_datetime(bar.index * MIN_NS, utc=True)
    return bar.sort_index()


def add_context(bar, sec_path="data/mbo/es_1s_v2.parquet"):
    """Wick location of the print, and how balanced the book was."""
    rng = (bar["high"] - bar["low"]).replace(0, np.nan)
    # 0 = at the low, 1 = at the high. A buy print at the high or a sell print
    # at the low is "at the wick" in the sense the framework means.
    pos = (bar["print_price"] - bar["low"]) / rng
    bar = bar.copy()
    bar["wick_pos"] = pos
    bar["at_extreme"] = np.where(
        bar["print_side"] > 0, pos >= 0.8,
        np.where(bar["print_side"] < 0, pos <= 0.2, False))

    if os.path.exists(sec_path):
        sec = pd.read_parquet(sec_path)
        qi = sec["qi"].resample("1min").last()
        bar["qi"] = qi.reindex(bar.index, method="ffill")
    else:
        bar["qi"] = np.nan
    bar["balanced"] = bar["qi"].abs() <= 0.3
    return bar


def rth_mask(idx):
    t = pd.DatetimeIndex(idx).tz_convert("UTC")
    m = t.hour * 60 + t.minute
    return (m >= 13 * 60 + 30) & (m < 20 * 60)


def forward(sub, minutes, full=None, same_session=True):
    """Signed P&L in points of holding the print's direction for `minutes`.

    `sub` is the selection being measured; `full` is the complete minute table
    the exit price is read from. That separation is the whole point: every
    selection here is sparse -- minutes containing a print above some size --
    so looking the exit up inside the selection asks for a minute that is
    almost never in it. Doing that silently reduced five hundred trades to ten,
    and shifting by rows instead answers "twenty prints later", not twenty
    minutes.

    Requiring entry and exit to fall in the same regular session handles the
    rest. `NQ.c.0` is a continuous contract that rolls quarterly, gapping
    hundreds of points overnight -- thousands of dollars inside any window
    spanning it, enough for two of them to swamp five hundred real trades.
    Rolls, weekend gaps and the overnight break all sit outside RTH, so a
    window contained in one session cannot contain one.
    """
    if full is None:
        full = sub
    close = full["close"]
    exit_ts = pd.DatetimeIndex(sub.index) + pd.Timedelta(minutes=minutes)
    exit_px = pd.Series(close.reindex(exit_ts).to_numpy(), index=sub.index)
    pnl = sub["print_side"] * (exit_px - close.reindex(sub.index).to_numpy())

    if same_session:
        ok = (rth_mask(sub.index) & rth_mask(exit_ts)
              & (pd.DatetimeIndex(sub.index).normalize()
                 == exit_ts.normalize()))
        pnl = pnl.where(ok)
    return pnl


def summarise(pnl, point_value, label, cost_ticks=1.0, tick=0.25):
    """Average P&L per trade in dollars, before and after crossing the spread."""
    p = pnl.dropna()
    if len(p) < 30:
        return {"label": label, "n": len(p)}
    gross = p * point_value
    cost = cost_ticks * tick * point_value
    net = gross - cost
    wins = net[net > 0].sum()
    losses = -net[net < 0].sum()
    return {"label": label, "n": len(p),
            "gross": float(gross.mean()), "net": float(net.mean()),
            "pf": float(wins / losses) if losses > 0 else np.nan,
            "hit": float((net > 0).mean()),
            "t": (float(net.mean() / (net.std(ddof=1) / np.sqrt(len(net))))
                  if net.std(ddof=1) > 0 else np.nan)}


def fmt(rows):
    hdr = (f"{'bucket':<26s}{'n':>7s}{'gross $':>10s}{'net $':>10s}"
           f"{'PF':>7s}{'hit%':>7s}{'t':>7s}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        if r.get("n", 0) < 30:
            out.append(f"{r['label']:<26s}{r.get('n',0):>7d}   (too few)")
            continue
        out.append(f"{r['label']:<26s}{r['n']:>7d}{r['gross']:>10.2f}"
                   f"{r['net']:>10.2f}{r['pf']:>7.2f}{r['hit']*100:>7.1f}{r['t']:>7.2f}")
    return "\n".join(out)


def trades_from_trades_file(path, progress_every=5_000_000):
    """Same output as `trades_from_mbo`, from the far cheaper `trades` schema.

    His framework needs the largest print in each minute and the state of the
    book, not the full order-level feed. `trades` plus `bbo-1s` carries both at
    a fraction of MBO's size, which is what makes a six-month sample affordable.
    """
    import databento as db
    store = db.DBNStore.from_file(path)
    ts, px, sz, sd = [], [], [], []
    n = 0
    for r in store:
        ts.append(r.ts_event)
        px.append(r.price)
        sz.append(r.size)
        sd.append(1 if r.side == "B" else (-1 if r.side == "A" else 0))
        n += 1
        if progress_every and n % progress_every == 0:
            print(f"  {n:,} trades", flush=True)
    return pd.DataFrame({"ts": ts, "price": np.array(px) / PRICE_SCALE,
                         "size": sz, "side": sd})


def qi_from_bbo(path, progress_every=5_000_000):
    """Per-second queue imbalance at the top of book, from a bbo-1s file."""
    import databento as db
    store = db.DBNStore.from_file(path)
    ts, qi = [], []
    n = 0
    for r in store:
        b, a = r.bid_sz_00, r.ask_sz_00
        tot = b + a
        if tot <= 0:
            continue
        ts.append(r.ts_event)
        qi.append((b - a) / tot)
        n += 1
        if progress_every and n % progress_every == 0:
            print(f"  {n:,} quotes", flush=True)
    s = pd.Series(qi, index=pd.to_datetime(ts, utc=True)).sort_index()
    return s[~s.index.duplicated(keep="last")]
