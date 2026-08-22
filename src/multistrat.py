"""Run the sleeves simultaneously and report what the combination is worth.

Sleeves are combined by inverse trailing volatility, computed on a rolling
window and lagged, so the allocation on any day uses only volatility that had
already been observed.  A cap keeps a quiet sleeve from being levered into
dominance by a short calm stretch.

Two things are reported that a single equity curve hides: the correlation
between sleeves, which is where the diversification actually comes from, and
each sleeve's contribution to the combined return, which is usually far less
even than the weights suggest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def periods_per_year(idx) -> float:
    """Observations per year implied by the data, not assumed.

    The sleeves sit on a union index that includes gold's Sunday session, so it
    carries about 312 rows a year rather than 252. Assuming 252 would stretch
    the elapsed time by a quarter and quietly understate every CAGR.
    """
    idx = pd.DatetimeIndex(idx)
    if len(idx) < 2:
        return float(TRADING_DAYS)
    span = (idx[-1] - idx[0]).days / 365.25
    return len(idx) / span if span > 0 else float(TRADING_DAYS)


def stats(r: pd.Series, rf=0.0) -> dict:
    r = r.dropna()
    if len(r) < 20:
        return {"n": len(r)}
    cum = (1 + r).cumprod()
    ppy = periods_per_year(r.index)
    years = (pd.DatetimeIndex(r.index)[-1] - pd.DatetimeIndex(r.index)[0]).days / 365.25
    cagr = cum.iloc[-1] ** (1 / years) - 1 if years > 0 and cum.iloc[-1] > 0 else np.nan
    vol = r.std() * np.sqrt(ppy)
    dd = (cum / cum.cummax() - 1)
    downside = r[r < 0].std() * np.sqrt(ppy)
    t = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else np.nan
    return {
        "n": len(r), "years": years, "ppy": ppy, "cagr": cagr, "vol": vol,
        "sharpe": (r.mean() * ppy - rf) / vol if vol > 0 else np.nan,
        "sortino": (r.mean() * ppy - rf) / downside if downside > 0 else np.nan,
        "maxdd": dd.min(), "calmar": cagr / abs(dd.min()) if dd.min() < 0 else np.nan,
        "hit": float((r > 0).mean()), "t": t,
        "total": cum.iloc[-1] - 1,
        "active": float((r.abs() > 1e-12).mean()),
    }


def fmt_stats(rows: list[tuple[str, dict]]) -> str:
    hdr = (f"{'sleeve':<26s}{'days':>7s}{'CAGR':>8s}{'vol':>7s}{'Sharpe':>8s}"
           f"{'Sortino':>9s}{'maxDD':>8s}{'Calmar':>8s}{'t':>7s}{'active':>8s}")
    out = [hdr, "-" * len(hdr)]
    for name, s in rows:
        if s.get("n", 0) < 20:
            out.append(f"{name:<26s}{s.get('n', 0):>7d}   (insufficient)")
            continue
        out.append(
            f"{name:<26s}{s['n']:>7d}{s['cagr']*100:>7.2f}%{s['vol']*100:>6.1f}%"
            f"{s['sharpe']:>8.2f}{s['sortino']:>9.2f}{s['maxdd']*100:>7.1f}%"
            f"{s['calmar']:>8.2f}{s['t']:>7.2f}{s['active']*100:>7.0f}%")
    return "\n".join(out)


def _plain_dates(idx) -> pd.DatetimeIndex:
    """Calendar dates with no timezone, so sleeves on different clocks align.

    Sleeves are built in different timezones (NY sessions, UTC bars, exchange
    dates). Comparing them requires one common key, and the calendar date is
    the only one that means the same thing to all of them.
    """
    i = pd.DatetimeIndex(idx)
    if i.tz is not None:
        i = i.tz_localize(None)
    return i.normalize()


def align(sleeves: dict[str, pd.Series]) -> pd.DataFrame:
    """One row per calendar day any sleeve traded; non-trading days are zero.

    A sleeve that is flat earns nothing rather than dropping the day, which is
    what makes the combined series a real account curve instead of a
    concatenation of active periods.
    """
    idx = None
    for s in sleeves.values():
        i = _plain_dates(s.index)
        idx = i if idx is None else idx.union(i)
    idx = idx.sort_values()
    out = {}
    for k, s in sleeves.items():
        s = s.copy()
        s.index = _plain_dates(s.index)
        s = s[~s.index.duplicated(keep="last")]
        out[k] = s.reindex(idx).fillna(0.0)
    return pd.DataFrame(out, index=idx)


def inverse_vol_weights(frame: pd.DataFrame, window=126, cap=4.0, min_obs=60):
    """Rolling inverse-vol allocation, lagged so no day uses its own vol.

    Sleeves that trade rarely (the January one) would show a near-zero
    volatility over a window in which they were flat, so volatility is measured
    over active days only and the resulting weight is capped.
    """
    active = frame.abs() > 1e-12
    vol = pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    for c in frame.columns:
        # Roll over the sleeve's own active days, not calendar days. A sleeve
        # that trades in bursts (January gold: ~21 days a year) would never
        # accumulate `min_obs` observations inside a calendar window, so its
        # volatility would stay NaN and it would silently receive no weight.
        s = frame[c][active[c]]
        v = s.rolling(window, min_periods=min_obs).std()
        vol[c] = v.reindex(frame.index).ffill()
    vol = vol.shift(1)
    inv = 1.0 / vol.replace(0.0, np.nan)
    inv = inv.div(inv.median(axis=0), axis=1).clip(upper=cap)
    w = inv.div(inv.sum(axis=1), axis=0)
    return w.fillna(0.0)


def combine(frame: pd.DataFrame, mode="invvol", target_vol=None, **kw):
    """Combined daily return series under a given allocation rule."""
    if mode == "equal":
        w = pd.DataFrame(1.0 / frame.shape[1], index=frame.index,
                         columns=frame.columns)
    elif mode == "invvol":
        w = inverse_vol_weights(frame, **kw)
    else:
        raise ValueError(mode)
    r = (frame * w).sum(axis=1)
    if target_vol:
        ppy = periods_per_year(r.index)
        realised = r.rolling(126, min_periods=60).std().shift(1) * np.sqrt(ppy)
        lev = (target_vol / realised).clip(upper=3.0).fillna(1.0)
        r = r * lev
    return r, w


def contribution(frame: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """Each sleeve's share of the combined total return."""
    c = (frame * w).sum(axis=0)
    return (c / c.abs().sum()).sort_values(ascending=False)


def drawdown_table(r: pd.Series, top=5):
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    out, d = [], dd.copy()
    for _ in range(top):
        if d.min() >= 0:
            break
        end = d.idxmin()
        start = cum.loc[:end].idxmax()
        rec = dd.loc[end:]
        recovered = rec[rec >= -1e-9]
        back = recovered.index[0] if len(recovered) else None
        out.append((start, end, back, float(dd.loc[end])))
        d.loc[start:(back if back is not None else d.index[-1])] = 0.0
    return out


def benchmarks(start="2015-01-01", path="data/yahoo"):
    """Passive reference points over the same window.

    Without these a Sharpe ratio has no scale. Buy-and-hold equities and a
    60/40 are what the strategy has to beat to be worth running, and over a
    sample dominated by an equity bull market they are a hard bar.
    """
    import os
    out = {}
    px = {}
    for sym in ("SPY", "IEF"):
        f = os.path.join(path, f"{sym}.parquet")
        if not os.path.exists(f):
            return out
        d = pd.read_parquet(f)
        s = d.set_index(pd.DatetimeIndex(d["date"]).tz_localize(None))["adj_close"]
        px[sym] = s[s.index >= start]

    out["SPY buy and hold"] = stats(px["SPY"].pct_change().dropna())
    out["IEF buy and hold"] = stats(px["IEF"].pct_change().dropna())
    both = pd.DataFrame(px).dropna()
    out["60/40 SPY/IEF"] = stats((both.pct_change() * [0.6, 0.4]).sum(axis=1).dropna())
    return out
