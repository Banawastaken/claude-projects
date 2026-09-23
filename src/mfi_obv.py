"""Money Flow Index against On Balance Volume, and whether their gap pays.

The idea as described: MFI leads, OBV lags, and when a wide gap opens between
them OBV closes it, so the gap points at where price is going.

There is a measurement problem to settle first. MFI is bounded to 0-100. OBV
is a running sum of signed volume -- the screenshot reads 545,114 -- and is
unbounded, in units of contracts. Two series that differ by four orders of
magnitude cannot have a gap between them until something puts them on one
scale, and what does that on a chart is the platform, which rescales each
indicator to whatever is on screen. So the gap is a function of the window
being looked at: scroll, and it changes. Worse, a min-max rescaled series
cannot stay at the top or bottom of its own range, so "OBV always closes the
gap" is partly true by construction rather than by market behaviour.

That does not kill the idea, it just means the normalisation has to be pinned
down and made causal before anything can be measured. OBV is put on 0-100 by
rolling min-max over a fixed lookback, which is what the chart does except
that the window is declared and only ever looks backwards. The window is a
parameter, so results are reported across all of them rather than at the one
that looks best.

The second thing to separate is convergence from profit. The gap can close
because OBV rises, or because MFI falls, and only the first has anything to
do with price. So the spread's own mean reversion and the spread's ability to
predict forward price return are tested separately -- the first is nearly
guaranteed, the second is the only one that can be traded.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def mfi(df: pd.DataFrame, n: int = 21) -> pd.Series:
    """Money Flow Index: RSI computed on price times volume rather than price."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    raw = tp * df["volume"]
    up = raw.where(tp > tp.shift(1), 0.0)
    dn = raw.where(tp < tp.shift(1), 0.0)
    pos = up.rolling(n).sum()
    neg = dn.rolling(n).sum()
    # A window with no down-ticks divides by zero; that is a maximal reading,
    # not a missing one.
    out = 100.0 - 100.0 / (1.0 + pos / neg.replace(0.0, np.nan))
    return out.where(neg > 0, 100.0).where(pos > 0, out.where(pos > 0, 0.0))


def obv(df: pd.DataFrame) -> pd.Series:
    """Cumulative signed volume: add on an up close, subtract on a down close."""
    d = np.sign(df["close"].diff().fillna(0.0))
    return (d * df["volume"]).cumsum()


def roll_norm(s: pd.Series, n: int) -> pd.Series:
    """Rescale to 0-100 by rolling min-max -- a chart autoscale with a fixed,
    backward-looking window. Flat windows have no range to scale into and are
    left undefined rather than forced to a midpoint."""
    lo = s.rolling(n).min()
    hi = s.rolling(n).max()
    rng = hi - lo
    return (100.0 * (s - lo) / rng.where(rng > 0)).astype(float)


def features(df: pd.DataFrame, n_mfi: int = 21, n_norm: int = 100) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["mfi"] = mfi(df, n_mfi)
    out["obv_n"] = roll_norm(obv(df), n_norm)
    out["spread"] = out["mfi"] - out["obv_n"]
    return out


def forward_return(close: pd.Series, k: int) -> pd.Series:
    """Return from this bar's close to the close k bars later."""
    return close.shift(-k) / close - 1.0


def load_h1(sym: str, path="data/decade") -> pd.DataFrame:
    d = pd.read_parquet(os.path.join(path, f"{sym}.parquet"))
    d = d.set_index(pd.DatetimeIndex(d["ts"]).tz_convert(None)).sort_index()
    return d[["open", "high", "low", "close", "volume", "spread_med"]]


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """M1 bars up to any coarser bar, keeping volume additive."""
    g = df.resample(rule, label="right", closed="right")
    out = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    if "spread_med" in df:
        out["spread_med"] = g["spread_med"].median()
    return out.dropna(subset=["close"])


def trades(df, spread, thr, hold, cost=None):
    """Non-overlapping trades, charged the real spread on entry and on exit.

    Sampling a signal on every bar and averaging the forward return counts one
    move many times over and charges one leg of cost instead of two, which
    flatters a short-horizon idea badly. Here a position is opened only when
    flat, held for a fixed number of bars, and billed both ways.
    """
    if cost is None:
        cost = (df["spread_med"] / df["close"]).fillna(0.0)
    side = np.where(spread > thr, 1.0, np.where(spread < -thr, -1.0, 0.0))
    fwd = forward_return(df["close"], hold)
    rows, i, n, idx = [], 0, len(df), df.index
    while i < n - hold:
        if side[i] != 0 and np.isfinite(fwd.iloc[i]) and np.isfinite(spread.iloc[i]):
            rows.append({"ts": idx[i], "side": side[i], "gross": side[i] * fwd.iloc[i],
                         "cost": cost.iloc[i] + cost.iloc[min(i + hold, n - 1)]})
            i += hold
        else:
            i += 1
    t = pd.DataFrame(rows)
    if not t.empty:
        t["net"] = t["gross"] - t["cost"]
    return t
