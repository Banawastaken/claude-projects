"""Each replicable MatFinOg strategy, as a daily net return stream.

Every sleeve emits the same thing: a pandas Series of daily returns at its
natural (unlevered) sizing, already net of its own trading costs.  Combining
strategies is then just combining series, and no sleeve can quietly borrow
another's leverage.

Two rules hold everywhere and are the ones that decide whether a backtest is
real:

  * signals are computed from data up to and including day t and applied to
    the return of day t+1.  Nothing reads its own future.
  * costs are charged on turnover -- the change in position -- not per
    "trade", so a sleeve that holds through a rebalance pays nothing and one
    that flips pays twice.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

NY = "America/New_York"

# Round-trip cost per unit of turnover, in basis points of notional, by the
# asset class the universe already assigns. Index CFDs and large ETFs sit near
# 2-4bp all-in at retail size; metals are the 1.96bp measured earlier in this
# project; crypto is far wider and must not be priced like a currency.
COST_BP = {"index": 3.0, "metal": 2.0, "etf": 3.0, "forex": 2.0,
           "energy": 5.0, "crypto": 25.0}
DEFAULT_COST_BP = 5.0


def cost_for(name: str) -> float:
    """Cost in bp for an instrument, from the universe's own asset class.

    Matching on substrings of the ticker was getting crypto wrong (BTCUSD was
    being charged 2bp, a currency's cost), so the class comes from the one
    place that already defines it.
    """
    from universe import UNIVERSE
    for inst in UNIVERSE:
        if inst.fn_name == name:
            return COST_BP.get(inst.asset_class, DEFAULT_COST_BP)
    return DEFAULT_COST_BP


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_decade_daily(name, path="data/decade"):
    """NY-daily OHLC from the H1 decade files, with a mid-price close."""
    f = os.path.join(path, f"{name}.parquet")
    if not os.path.exists(f):
        return None
    df = pd.read_parquet(f)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[(df["high"] > df["low"]) | (df["volume"] > 0)]
    if df.empty:
        return None
    df["date"] = df["ts"].dt.tz_convert(NY).dt.normalize()
    g = df.groupby("date")
    out = pd.DataFrame({
        "close": 0.5 * (g["close"].last() + g["ask_close"].last()),
        "high": g["high"].max(), "low": g["low"].min(),
        "open": 0.5 * (g["open"].first() + g["ask_open"].first()),
    })
    return out[out["close"] > 0]


def load_yahoo(sym, path="data/yahoo"):
    f = os.path.join(path, f"{sym}.parquet")
    if not os.path.exists(f):
        return None
    df = pd.read_parquet(f)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[["adj_close"]].rename(
        columns={"adj_close": "close"})


def _returns_from_weights(px_close, weight, cost_bp):
    """Daily net return from a weight series, executed with a one-day lag.

    `weight` is the target for day t, decided from information up to t.  It is
    shifted so it earns day t+1's return, and turnover is charged when the
    position actually changes.
    """
    r = px_close.pct_change()
    w = weight.reindex(px_close.index).ffill().fillna(0.0)
    held = w.shift(1).fillna(0.0)
    turn = held.diff().abs().fillna(held.abs())
    return (held * r - turn * cost_bp / 1e4).fillna(0.0)


# --------------------------------------------------------------------------
# sleeve 1 -- overnight index drift  ("pays while you sleep")
# --------------------------------------------------------------------------

def short_rate(path="data/yahoo/^IRX.parquet"):
    """Daily 13-week T-bill yield as a decimal, or None if not downloaded.

    A flat financing rate across 2015-2026 is not a small simplification: the
    bill yield was 0.06% in 2015 and 5.3% in 2023, so a single number either
    charges the zero-rate years for money that was free or lets the recent
    years finance too cheaply.
    """
    if not os.path.exists(path):
        return None
    d = pd.read_parquet(path)
    s = d.set_index(pd.DatetimeIndex(d["date"]).tz_localize(None))["close"] / 100.0
    return s.clip(lower=0.0)


def sleeve_overnight(names=("NDX100", "SPX500"), markup=0.020,
                     fallback_annual=0.065):
    """Long the index from the 16:00 NY close to the next 09:00 NY open.

    Uses the intraday legs rather than daily bars, so the return is the actual
    overnight window with the bid/ask round trip paid and one night of
    financing charged.  Financing is the prevailing 13-week bill yield plus a
    broker markup, per night, and falls back to a flat rate only if the rate
    history is missing.  Equal weight across the named indices.
    """
    import sys
    sys.path.insert(0, "src")
    from anomaly import financing_bp, load, overnight_legs

    rate = short_rate()
    cols = {}
    for n in names:
        df = load(n)
        on = overnight_legs(df)
        idx = pd.DatetimeIndex(pd.to_datetime(on["date"])).tz_localize(None)
        if rate is None:
            fin = pd.Series(financing_bp(fallback_annual), index=idx)
        else:
            r_ann = rate.reindex(idx, method="ffill").fillna(fallback_annual) + markup
            fin = r_ann / 360.0 * 1e4
        r = (pd.Series(on["net_bp"].values, index=idx) - fin) / 1e4
        cols[n] = r
    frame = pd.DataFrame(cols)
    return frame.mean(axis=1).dropna(), frame


# --------------------------------------------------------------------------
# sleeve 2 -- gold calendar seasonality
# --------------------------------------------------------------------------

def sleeve_gold_january(name="XAUUSD"):
    """Long gold through January, flat the rest of the year.

    The month is known in advance, so there is no signal lag to worry about --
    only the execution lag, which `_returns_from_weights` applies.
    """
    px = load_decade_daily(name)
    w = pd.Series((px.index.month == 1).astype(float), index=px.index)
    return _returns_from_weights(px["close"], w, cost_for(name)), px


# --------------------------------------------------------------------------
# sleeve 3 -- Faber tactical allocation
# --------------------------------------------------------------------------

FABER = ["SPY", "EFA", "IEF", "VNQ", "DBC"]


def sleeve_faber(symbols=FABER, months=10):
    """Hold each asset while it is above its 10-month moving average, else cash.

    Faber's rule as published: monthly close, monthly rebalance, equal weight
    across the five sleeves, cash earning nothing (a conservative simplification
    -- real cash earned the bill rate over most of this sample).
    """
    px = {}
    for s in symbols:
        d = load_yahoo(s)
        if d is not None:
            px[s] = d["close"]
    if not px:
        return None, None
    px = pd.DataFrame(px).dropna(how="all").ffill()

    monthly = px.resample("ME").last()
    sma = monthly.rolling(months).mean()
    sig = (monthly > sma).astype(float)
    # ffill already carries a month-end signal forward onto the days that
    # follow it, and `_returns_from_weights` adds the one-day execution lag.
    # An extra shift here would hold each month-end decision for a further
    # month: the February 2020 exit would not have traded until 1 April,
    # staying long through the entire COVID drawdown.
    daily_sig = sig.reindex(px.index, method="ffill")

    per = {}
    for s in px.columns:
        per[s] = _returns_from_weights(px[s], daily_sig[s] / len(px.columns),
                                       COST_BP["etf"])
    frame = pd.DataFrame(per)
    return frame.sum(axis=1).dropna(), frame


# --------------------------------------------------------------------------
# sleeve 4 -- time-series momentum
# --------------------------------------------------------------------------

def sleeve_tsmom(names, lookback_months=12, vol_window=60, target_vol=0.10):
    """Long or short each market on the sign of its 12-month return.

    Sized inverse to each market's own trailing volatility so a quiet currency
    and a loud index contribute comparably, which is how the published version
    of this strategy works and the only way the average means anything.
    """
    px = {}
    for n in names:
        d = load_decade_daily(n)
        if d is not None and len(d) > 400:
            px[n] = d["close"]
    if not px:
        return None, None
    px = pd.DataFrame(px).ffill()

    r = px.pct_change()
    lb = int(lookback_months * 21)
    signal = np.sign(px / px.shift(lb) - 1.0)
    vol = r.rolling(vol_window).std() * np.sqrt(252)
    scale = (target_vol / vol).clip(upper=3.0)
    w = (signal * scale) / max(len(px.columns), 1)
    # Rebalance monthly rather than daily: daily reweighting of a 12-month
    # signal is nearly all turnover and no information.
    w = w.resample("ME").last().reindex(px.index, method="ffill")

    per = {}
    for n in px.columns:
        per[n] = _returns_from_weights(px[n], w[n], cost_for(n))
    frame = pd.DataFrame(per)
    return frame.sum(axis=1).dropna(), frame


# --------------------------------------------------------------------------
# sleeve 5 -- post-earnings announcement drift
# --------------------------------------------------------------------------

def sleeve_pead(top=0.2, hold=60, min_adv=5e6):
    """Long the best earnings reactions, short the worst, held for a quarter.

    Returns are already market-adjusted and dollar-neutral inside `pead.run`,
    so this sleeve is a spread rather than a directional position, and its
    correlation with the others should be near zero for structural reasons
    rather than by luck.

    `min_adv` drops names too thin to trade at size. PEAD is strongest in
    exactly those names, so this deliberately gives up some of the measured
    effect in exchange for a number that could be realised.
    """
    import sys
    sys.path.insert(0, "src")
    try:
        from pead import run as pead_run
    except Exception:
        return None, None
    res = pead_run(top=top, hold=hold, min_adv=min_adv)
    if res is None:
        return None, None
    return res["ret"].dropna(), res
