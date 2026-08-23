"""Do the dealer gamma levels do what both channels say they do?

Three claims, each measured separately because they can fail independently.

  1. Sign of gamma sets the volatility regime. Dealers long gamma hedge against
     the move and damp it; short gamma, they chase and amplify it. This is the
     mechanical core -- if it is not visible, nothing built on top of it can be.

  2. Price reverses at the call and put walls.

  3. The gamma flip is a boundary price respects.

Levels for a session come from open interest and settlement prices stamped
before it opens, so everything below is what a trader could have drawn on the
chart at the open. Windows follow the project's convention: the first half is
looked at, the second half is held back.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LEVELS = "data/opra/ndx_gamma_levels.parquet"


def intraday(path="data/decade/NDX100.parquet"):
    """H1 NDX bars, keyed by UTC date."""
    df = pd.read_parquet(path)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[(df["high"] > df["low"]) | (df["volume"] > 0)].copy()
    df["date"] = df["ts"].dt.normalize()
    df["mid"] = 0.5 * (df["close"] + df["ask_close"])
    return df


def daily_from(bars):
    g = bars.groupby("date")
    return pd.DataFrame({
        "open": g["mid"].first(), "close": g["mid"].last(),
        "high": g["high"].max(), "low": g["low"].min(),
        "bars": g["mid"].size,
        # Realised volatility inside the session, which is what a gamma regime
        # is supposed to move.
        "rv": g["mid"].apply(lambda s: float(s.pct_change().std())),
    })


def load():
    lv = pd.read_parquet(LEVELS)
    lv.index = pd.DatetimeIndex(lv.index)
    d = daily_from(intraday())
    j = lv.join(d, how="inner", rsuffix="_px").dropna(subset=["close", "rv"])
    j["ret"] = j["close"] / j["open"] - 1.0
    j["range_pct"] = (j["high"] - j["low"]) / j["open"]
    return j


def summarise(x, label):
    x = pd.Series(x).dropna()
    if len(x) < 20:
        return {"label": label, "n": len(x)}
    m, s = float(x.mean()), float(x.std(ddof=1))
    return {"label": label, "n": len(x), "mean": m, "sd": s,
            "t": m / (s / np.sqrt(len(x))) if s > 0 else np.nan}


def fmt(rows, scale=1.0, unit=""):
    hdr = f"{'group':<34s}{'n':>6s}{'mean':>12s}{'t':>8s}"
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        if r.get("n", 0) < 20:
            out.append(f"{r['label']:<34s}{r.get('n',0):>6d}   (too few)")
            continue
        out.append(f"{r['label']:<34s}{r['n']:>6d}{r['mean']*scale:>11.3f}{unit}"
                   f"{r['t']:>8.2f}")
    return "\n".join(out)


def main():
    j = load()
    half = j.index[len(j) // 2]
    print(f"{len(j)} sessions, {j.index.min().date()} .. {j.index.max().date()}")
    print(f"design through {half.date()}, holdout after\n")

    print("=" * 62)
    print("1. GAMMA SIGN AND THE VOLATILITY REGIME")
    print("=" * 62)
    for tag, w in (("full sample", j), ("design", j[j.index <= half]),
                   ("holdout", j[j.index > half])):
        pos = w[w["total_gex"] > 0]
        neg = w[w["total_gex"] <= 0]
        rows = [summarise(pos["rv"], f"  {tag}: dealers long gamma"),
                summarise(neg["rv"], f"  {tag}: dealers short gamma")]
        print(fmt(rows, 1e4, "bp"))
        if len(pos) > 20 and len(neg) > 20:
            d = neg["rv"].mean() - pos["rv"].mean()
            se = np.sqrt(neg["rv"].var()/len(neg) + pos["rv"].var()/len(pos))
            print(f"    short minus long: {d*1e4:+.2f} bp, t = {d/se:+.2f}"
                  f"   (theory says positive)\n")

    print("=" * 62)
    print("2. DOES PRICE REVERSE AT THE WALLS?")
    print("=" * 62)
    # A level is only resistance if price arrives at it from below. Counting a
    # session whose call wall already sat under the open measures how far the
    # day closed above a level it never approached, which is how the first
    # version of this test produced +298bp and called it a reversal.
    up = j[(j["call_wall"] > j["open"]) & (j["high"] >= j["call_wall"])]
    dn = j[(j["put_wall"] < j["open"]) & (j["low"] <= j["put_wall"])]
    cand_up = j[j["call_wall"] > j["open"]]
    cand_dn = j[j["put_wall"] < j["open"]]
    print(f"  call wall above the open on {len(cand_up)} sessions; reached on "
          f"{len(up)} ({len(up)/max(len(cand_up),1)*100:.0f}%)")
    print(f"  put wall below the open on {len(cand_dn)} sessions; reached on "
          f"{len(dn)} ({len(dn)/max(len(cand_dn),1)*100:.0f}%)\n")
    rows = [
        summarise(up["close"] / up["call_wall"] - 1,
                  "  call wall reached -> close"),
        summarise(-(dn["close"] / dn["put_wall"] - 1),
                  "  put wall reached -> close (short)"),
    ]
    print(fmt(rows, 1e4, "bp"))
    print("    both are the P&L of fading the level to the close, so a working\n"
          "    wall gives a positive number here.\n")

    print("  against the same trade at a level that is not the wall:")
    for mult, name in ((1.005, "open +0.5%"), (1.010, "open +1.0%")):
        lvl = j["open"] * mult
        hit = j[j["high"] >= lvl]
        rows = [summarise(-(hit["close"] / (hit["open"] * mult) - 1),
                          f"  fade {name} -> close")]
        print(fmt(rows, 1e4, "bp"))
    print()

    print("=" * 62)
    print("3. THE GAMMA FLIP AS A BOUNDARY")
    print("=" * 62)
    f = j.dropna(subset=["gamma_flip"])
    above = f[f["open"] > f["gamma_flip"]]
    below = f[f["open"] <= f["gamma_flip"]]
    print(f"  flip located on {len(f)} of {len(j)} sessions")
    print(fmt([summarise(above["rv"], "  opened above the flip"),
               summarise(below["rv"], "  opened below the flip")], 1e4, "bp"))
    print()
    print(fmt([summarise(above["ret"], "  return, opened above flip"),
               summarise(below["ret"], "  return, opened below flip")], 1e4, "bp"))


if __name__ == "__main__":
    main()
