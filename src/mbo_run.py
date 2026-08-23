"""Aggregate a DBN top-of-book file into one-second features, then test them.

The file is streamed rather than loaded: a week of ES top-of-book is tens of
millions of records and would be tens of gigabytes as a DataFrame. Each record
contributes its order-flow increment to a one-second bucket and is then
discarded, so memory stays flat regardless of how much data is bought.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PRICE_SCALE = 1e9
NS = 1_000_000_000


def aggregate(path, bucket_ns=NS, progress_every=5_000_000):
    """Stream an mbp-1 DBN file into per-second OFI, queue imbalance and mid.

    OFI follows Cont, Kukanov & Stoikov: the bid side contributes its whole new
    size when the bid price rises, minus the previous size when it falls, and
    the size change when the price is unchanged; the ask is the mirror with the
    opposite sign.
    """
    import databento as db

    store = db.DBNStore.from_file(path)
    buckets = {}
    prev_bp = prev_bs = prev_ap = prev_as = None
    n = 0

    for r in store:
        bp, bs = r.bid_px_00, r.bid_sz_00
        ap, asz = r.ask_px_00, r.ask_sz_00
        # A one-sided book has no top-of-book imbalance to speak of.
        if bp <= 0 or ap <= 0 or bp > ap:
            continue
        n += 1
        if prev_bp is not None:
            demand = bs if bp > prev_bp else (-prev_bs if bp < prev_bp else bs - prev_bs)
            supply = asz if ap < prev_ap else (-prev_as if ap > prev_ap else asz - prev_as)
            e = demand - supply
            k = r.ts_event // bucket_ns
            b = buckets.get(k)
            tot = bs + asz
            qi = (bs - asz) / tot if tot else 0.0
            mid = (bp + ap) / 2.0 / PRICE_SCALE
            spread = (ap - bp) / PRICE_SCALE
            if b is None:
                buckets[k] = [e, qi, mid, 1, spread]
            else:
                b[0] += e
                b[1] = qi
                b[2] = mid
                b[3] += 1
                b[4] += spread
        prev_bp, prev_bs, prev_ap, prev_as = bp, bs, ap, asz
        if progress_every and n % progress_every == 0:
            print(f"  {n:,} quotes, {len(buckets):,} seconds", flush=True)

    if not buckets:
        return None
    ks = np.fromiter(buckets.keys(), dtype=np.int64)
    order = np.argsort(ks)
    ks = ks[order]
    vals = np.array([buckets[k] for k in ks], dtype=float)
    idx = pd.to_datetime(ks * bucket_ns, utc=True)
    out = pd.DataFrame({"ofi": vals[:, 0], "qi": vals[:, 1],
                        "mid": vals[:, 2], "updates": vals[:, 3],
                        "spread": vals[:, 4] / np.maximum(vals[:, 3], 1)},
                       index=idx)
    print(f"  streamed {n:,} quotes into {len(out):,} one-second bars")
    return out


def rth(df):
    """US regular trading hours, 13:30-20:00 UTC (09:30-16:00 ET, summer)."""
    t = df.index.tz_convert("UTC")
    return df[(t.hour * 60 + t.minute >= 13 * 60 + 30) &
              (t.hour * 60 + t.minute < 20 * 60)]


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else (
        "data/mbo/GLBX.MDP3_ES.c.0_mbp-1_2026-08-10_2026-08-17.dbn.zst")
    out = sys.argv[2] if len(sys.argv) > 2 else "data/mbo/es_1s_v2.parquet"
    if os.path.exists(out):
        feat = pd.read_parquet(out)
        print(f"cached {len(feat):,} bars")
    else:
        feat = aggregate(src)
        feat.to_parquet(out)
        print(f"wrote {out}")

    from mbo_features import fmt, predictive_test
    for label, d in (("all hours", feat), ("RTH only", rth(feat))):
        d = d[d["updates"] > 0]
        print(f"\n=== {label}: {len(d):,} one-second bars ===")
        print(fmt(predictive_test(d)))
