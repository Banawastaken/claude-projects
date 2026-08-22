"""Session and calendar anomaly studies on the decade H1 dataset.

This tests a different family of edge from the price-action work: returns
conditioned on *when* rather than on *what the chart did*.  Overnight drift,
day-of-week and month-of-year effects are all measured the same way here --
split the day into disjoint windows, hold one of them, and see what is left
after the costs of actually holding it.

Two costs matter and both are applied explicitly:
  * the bid/ask round trip, taken from the feed rather than assumed;
  * overnight financing, which is the whole game for an index CFD held
    through the night -- brokers price it to claw back exactly the drift
    this anomaly is trying to harvest.

Sessions are resolved in America/New_York so US daylight saving is handled
rather than smeared across the year.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"


def load(name: str, path="data/decade") -> pd.DataFrame:
    df = pd.read_parquet(f"{path}/{name}.parquet")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    # Zero-range filler bars carry no information and distort every average.
    live = (df["high"] > df["low"]) | (df["volume"] > 0)
    df = df[live].reset_index(drop=True)
    df["ny"] = df["ts"].dt.tz_convert(NY)
    df["ny_hour"] = df["ny"].dt.hour
    df["ny_date"] = df["ny"].dt.normalize()
    return df


def hour_bar(df: pd.DataFrame, hour: int) -> pd.DataFrame:
    """One row per NY date: the bar starting at `hour` NY time."""
    sel = df[df["ny_hour"] == hour]
    return sel.drop_duplicates("ny_date", keep="first").set_index("ny_date")


def overnight_legs(df, close_hour=15, open_hour=9):
    """Returns for the overnight hold and the cash session that follows it.

    `close_hour` is the last cash-session bar (15 NY = the 15:00-16:00 bar, so
    entry is at 16:00 NY).  `open_hour` is the bar whose open is the exit.

    Both legs are priced as a long would actually trade them: in at the ask,
    out at the bid.  Returns are simple, expressed in basis points.
    """
    c = hour_bar(df, close_hour)
    o = hour_bar(df, open_hour)

    # The overnight leg spans a date boundary, so pair each close with the
    # next available cash open strictly after it.
    c = c[["close", "ask_close", "ts"]].rename(
        columns={"close": "bid_c", "ask_close": "ask_c", "ts": "ts_c"})
    o = o[["open", "ask_open", "ts", "close", "ask_close"]].rename(
        columns={"open": "bid_o", "ask_open": "ask_o", "ts": "ts_o",
                 "close": "bid_oc", "ask_close": "ask_oc"})

    pair = pd.merge_asof(
        o.sort_values("ts_o"), c.sort_values("ts_c"),
        left_on="ts_o", right_on="ts_c", direction="backward",
        tolerance=pd.Timedelta("36h"))
    pair = pair.dropna(subset=["ts_c"])
    # Guard against pairing an open with a close on the same session.
    pair = pair[pair["ts_o"] > pair["ts_c"]]

    mid_c = 0.5 * (pair["bid_c"] + pair["ask_c"])
    mid_o = 0.5 * (pair["bid_o"] + pair["ask_o"])
    pair["gross_bp"] = (mid_o / mid_c - 1.0) * 1e4
    # long: pay the ask at the 16:00 entry, receive the bid at the exit
    pair["net_bp"] = (pair["bid_o"] / pair["ask_c"] - 1.0) * 1e4
    pair["spread_bp"] = ((pair["ask_c"] - pair["bid_c"]) / mid_c
                         + (pair["ask_o"] - pair["bid_o"]) / mid_o) * 1e4
    pair["date"] = pair["ts_o"].dt.tz_convert(NY).dt.normalize()
    pair["dow"] = pair["ts_o"].dt.tz_convert(NY).dt.dayofweek
    pair["month"] = pair["ts_o"].dt.tz_convert(NY).dt.month
    return pair.reset_index(drop=True)


def intraday_leg(df, open_hour=9, close_hour=15):
    """Cash-session return: open of `open_hour` bar to close of `close_hour` bar."""
    o = hour_bar(df, open_hour)[["open", "ask_open", "ts"]]
    c = hour_bar(df, close_hour)[["close", "ask_close"]]
    j = o.join(c, how="inner").dropna()
    mid_o = 0.5 * (j["open"] + j["ask_open"])
    mid_c = 0.5 * (j["close"] + j["ask_close"])
    return pd.DataFrame({
        "date": j.index,
        "gross_bp": (mid_c / mid_o - 1.0) * 1e4,
        "net_bp": (j["close"] / j["ask_open"] - 1.0) * 1e4,
    }).reset_index(drop=True)


def financing_bp(annual_rate=0.065, nights=1):
    """Cost of carrying a long index CFD overnight, in basis points of notional.

    Brokers quote long index financing as (reference rate + markup)/360 per
    calendar night.  The default is a mid-2020s SOFR plus a typical retail
    markup; it is a parameter because it is the single number this anomaly is
    most sensitive to.
    """
    return annual_rate / 360.0 * nights * 1e4


def summarise(x: pd.Series, label: str) -> dict:
    x = x.dropna()
    n = len(x)
    if n < 2:
        return {"label": label, "n": n}
    mu, sd = x.mean(), x.std(ddof=1)
    t = mu / (sd / np.sqrt(n)) if sd > 0 else np.nan
    # Annualise assuming one observation per trading day.
    ann = mu / 1e4 * 252
    return {"label": label, "n": n, "mean_bp": mu, "sd_bp": sd, "t": t,
            "hit": float((x > 0).mean()), "ann_pct": ann * 100,
            "sharpe": (mu / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "total_pct": float(x.sum() / 1e4 * 100)}


def fmt(rows) -> str:
    hdr = (f"{'window':<34s}{'n':>7s}{'mean bp':>10s}{'t':>7s}"
           f"{'hit%':>7s}{'ann%':>9s}{'Sharpe':>8s}{'total%':>10s}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        if r.get("n", 0) < 2:
            out.append(f"{r['label']:<34s}{r.get('n', 0):>7d}   (insufficient)")
            continue
        out.append(f"{r['label']:<34s}{r['n']:>7d}{r['mean_bp']:>10.2f}"
                   f"{r['t']:>7.2f}{r['hit']*100:>7.1f}{r['ann_pct']:>9.1f}"
                   f"{r['sharpe']:>8.2f}{r['total_pct']:>10.1f}")
    return "\n".join(out)
