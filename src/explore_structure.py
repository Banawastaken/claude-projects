"""Look for durable structure in gold, using design-window data only.

Rather than guessing at chart patterns, measure the statistical shape of the
market: which hours carry return, whether moves persist or revert at each
horizon, and how that differs by session. The design window is split in half so
an effect has to show up in 2015-2017 AND 2018-2020 to be worth pursuing --
an internal consistency check that costs nothing and does not touch the
held-out years.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DECADE = os.path.join(HERE, "..", "data", "instruments", "XAUUSD_H1_DECADE.parquet")

HALVES = [("2015-2017", 2015, 2017), ("2018-2020", 2018, 2020)]


def load():
    df = pd.read_parquet(DECADE).sort_values("ts").reset_index(drop=True)
    df["year"] = df["ts"].dt.year
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek
    df["ret"] = df["close"].pct_change() * 10000  # basis points
    df["absret"] = df["ret"].abs()
    return df.dropna(subset=["ret"])


def by_group(df, col, label):
    print(f"\n--- mean H1 return (bp) by {label}, and whether it holds in both halves")
    rows = []
    for key, g in df.groupby(col):
        rec = {label: key, "n": len(g)}
        for name, y0, y1 in HALVES:
            sub = g[(g["year"] >= y0) & (g["year"] <= y1)]
            rec[name] = sub["ret"].mean() if len(sub) > 30 else np.nan
        rec["both"] = ("yes" if (np.sign(rec[HALVES[0][0]]) == np.sign(rec[HALVES[1][0]])
                                 and abs(rec[HALVES[0][0]]) > 0.3
                                 and abs(rec[HALVES[1][0]]) > 0.3) else "")
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["mean"] = out[[h[0] for h in HALVES]].mean(axis=1)
    print(out.round(2).to_string(index=False))
    return out


def persistence(df):
    """Does an up move predict the next move, or reverse it?

    Positive correlation means momentum at that horizon, negative means
    reversal. Measured separately in each half of the design window.
    """
    print("\n--- return autocorrelation at several horizons (design window halves)")
    print(f"{'horizon':>10s} " + " ".join(f"{h[0]:>12s}" for h in HALVES) + f" {'agree':>7s}")
    close = df.set_index("ts")["close"]
    for label, bars in [("1h", 1), ("4h", 4), ("12h", 12), ("1d", 24),
                        ("3d", 72), ("1w", 120)]:
        vals = []
        for name, y0, y1 in HALVES:
            c = close[(close.index.year >= y0) & (close.index.year <= y1)]
            r = np.log(c).diff(bars).dropna()
            if len(r) < 200:
                vals.append(np.nan)
                continue
            a = r.values[:-bars]
            b = r.values[bars:]
            vals.append(float(np.corrcoef(a, b)[0, 1]))
        agree = "yes" if (len(vals) == 2 and np.sign(vals[0]) == np.sign(vals[1])
                          and min(abs(v) for v in vals) > 0.02) else ""
        print(f"{label:>10s} " + " ".join(f"{v:12.4f}" for v in vals) + f" {agree:>7s}")


def overnight_vs_intraday(df):
    """Gold's return split by session, a structural rather than pattern effect."""
    print("\n--- return by session block (bp per bar), design halves")
    blocks = {"Asia 00-07": (0, 7), "London 07-13": (7, 13),
              "NY overlap 13-17": (13, 17), "NY late 17-21": (17, 21),
              "Close 21-24": (21, 24)}
    rows = []
    for name, (a, b) in blocks.items():
        g = df[(df["hour"] >= a) & (df["hour"] < b)]
        rec = {"block": name, "n": len(g)}
        for hname, y0, y1 in HALVES:
            sub = g[(g["year"] >= y0) & (g["year"] <= y1)]
            rec[hname] = sub["ret"].mean() if len(sub) > 30 else np.nan
        rec["vol"] = g["absret"].mean()
        rows.append(rec)
    out = pd.DataFrame(rows)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    df = load()
    design = df[(df["year"] >= 2015) & (df["year"] <= 2020)]
    print(f"design window: {len(design):,} H1 bars, "
          f"{design['ts'].min().date()} -> {design['ts'].max().date()}")

    by_group(design, "hour", "hour (UTC)")
    by_group(design, "dow", "weekday (0=Mon)")
    persistence(design)
    overnight_vs_intraday(design)
