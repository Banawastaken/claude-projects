"""Order-flow features, and a test of whether they predict anything.

The headline feature is order flow imbalance (Cont, Kukanov & Stoikov, 2014):
the net pressure on the book from size added and removed at the best quotes.
Their result is that OFI is close to linear in the contemporaneous price change
and carries some predictive power at short horizons.

A budget note that matters more than any parameter here: OFI needs only the top
of book, so it is computable from `mbp-1`, which is one or two orders of
magnitude smaller than full `mbo`. On a fixed trial credit that is the
difference between a few sessions and many months of history. Full MBO buys
queue-position detail -- where in the queue an order sits, how often it is
cancelled -- which is a genuinely different and richer signal, but not the one
to spend the whole budget on first.

Nothing here needs a Databento key: the maths runs on any frame with bid/ask
price and size columns, and is tested on synthetic books.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ofi(book: pd.DataFrame, bid_px="bid_px_00", bid_sz="bid_sz_00",
        ask_px="ask_px_00", ask_sz="ask_sz_00") -> pd.Series:
    """Per-update order flow imbalance at the best quotes.

    For each book update, the bid side contributes its full new size when the
    bid price rises (fresh demand), minus the previous size when it falls (demand
    pulled), and the size change when the price is unchanged. The ask side is the
    mirror, entering with the opposite sign.
    """
    bp = book[bid_px].to_numpy(float)
    bs = book[bid_sz].to_numpy(float)
    ap = book[ask_px].to_numpy(float)
    asz = book[ask_sz].to_numpy(float)

    e = np.zeros(len(book))
    bp0, bs0, ap0, as0 = bp[:-1], bs[:-1], ap[:-1], asz[:-1]
    bp1, bs1, ap1, as1 = bp[1:], bs[1:], ap[1:], asz[1:]

    demand = np.where(bp1 > bp0, bs1, np.where(bp1 < bp0, -bs0, bs1 - bs0))
    supply = np.where(ap1 < ap0, as1, np.where(ap1 > ap0, -as0, as1 - as0))
    e[1:] = demand - supply
    return pd.Series(e, index=book.index, name="ofi")


def queue_imbalance(book: pd.DataFrame, bid_sz="bid_sz_00",
                    ask_sz="ask_sz_00") -> pd.Series:
    """(bid - ask) / (bid + ask) at the top of book, in [-1, 1]."""
    b = book[bid_sz].to_numpy(float)
    a = book[ask_sz].to_numpy(float)
    tot = b + a
    return pd.Series(np.where(tot > 0, (b - a) / np.where(tot > 0, tot, 1), 0.0),
                     index=book.index, name="queue_imbalance")


def mid(book: pd.DataFrame, bid_px="bid_px_00", ask_px="ask_px_00") -> pd.Series:
    return (book[bid_px] + book[ask_px]) / 2.0


def resample_features(book: pd.DataFrame, freq="1s") -> pd.DataFrame:
    """Aggregate event-time book updates onto a clock.

    OFI sums over the interval; queue imbalance is a state, so it is taken at
    the interval's last update.
    """
    f = pd.DataFrame({"ofi": ofi(book), "qi": queue_imbalance(book),
                      "mid": mid(book)}, index=book.index)
    g = f.resample(freq)
    out = pd.DataFrame({"ofi": g["ofi"].sum(), "qi": g["qi"].last(),
                        "mid": g["mid"].last()}).dropna()
    return out


def predictive_test(feat: pd.DataFrame, horizons=(1, 5, 10, 30, 60)):
    """Does the feature at t explain the return after t, or only alongside it?

    The contemporaneous column is the sanity check -- OFI is known to be nearly
    linear in the same-interval move, so a near-zero there means the feature is
    being computed wrongly. The forward columns are the only ones that could be
    traded.
    """
    r = feat["mid"].pct_change().shift(-1)
    rows = []
    for h in horizons:
        fwd = feat["mid"].shift(-h) / feat["mid"] - 1.0
        same = feat["mid"] / feat["mid"].shift(1) - 1.0
        for name in ("ofi", "qi"):
            x = feat[name]
            d = pd.concat([x, fwd, same], axis=1).dropna()
            if len(d) < 100:
                continue
            xv = d.iloc[:, 0].to_numpy()
            fv = d.iloc[:, 1].to_numpy()
            sv = d.iloc[:, 2].to_numpy()
            rows.append({
                "feature": name, "horizon": h, "n": len(d),
                "corr_forward": float(np.corrcoef(xv, fv)[0, 1]),
                "corr_same": float(np.corrcoef(xv, sv)[0, 1]),
                "t_forward": float(np.corrcoef(xv, fv)[0, 1] *
                                   np.sqrt(max(len(d) - 2, 1)) /
                                   np.sqrt(max(1 - np.corrcoef(xv, fv)[0, 1] ** 2, 1e-12))),
            })
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame) -> str:
    hdr = f"{'feature':<10s}{'horizon':>9s}{'n':>9s}{'corr fwd':>11s}{'t fwd':>9s}{'corr same':>12s}"
    out = [hdr, "-" * len(hdr)]
    for r in df.itertuples():
        out.append(f"{r.feature:<10s}{r.horizon:>9d}{r.n:>9,d}"
                   f"{r.corr_forward:>11.4f}{r.t_forward:>9.2f}{r.corr_same:>12.4f}")
    return "\n".join(out)
