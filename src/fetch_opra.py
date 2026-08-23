"""Historical NDX option chains from OPRA: the input dealer gamma has always needed.

Free chains are a snapshot, which is why every gamma question so far has ended
at "no history". OPRA has the three pieces that reconstruct a dated chain:

  definition  strike, expiry and right for every contract
  statistics  end-of-day open interest, which is the whole weight in a GEX sum
  ohlcv-1d    the settlement price, from which implied volatility is inverted

Gamma is then computed rather than taken from a vendor, exactly as gex.py does
on the live chain, so the historical and live numbers are the same measurement.
"""

from __future__ import annotations

import os
import sys

OUT = "data/opra"
DATASET = "OPRA.PILLAR"


def client():
    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        raise SystemExit("DATABENTO_API_KEY is not set")
    import databento as db
    return db.Historical(key)


def plan(schema, start, end, parent="NDX.OPT"):
    c = client()
    kw = dict(dataset=DATASET, symbols=[parent], stype_in="parent",
              schema=schema, start=start, end=end)
    return float(c.metadata.get_cost(**kw)), int(c.metadata.get_billable_size(**kw))


def month_edges(start, end):
    """Month boundaries covering [start, end).

    A year of OPRA in one request times the gateway out at 504 -- the whole
    year has to be materialised before the first byte is sent. Month-sized
    requests stream fine and cost the same, since billing is by the byte
    delivered, and a failure loses one month rather than the lot.
    """
    import datetime as dt
    a = dt.date.fromisoformat(start)
    b = dt.date.fromisoformat(end)
    out = []
    while a < b:
        nxt = (a.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        out.append((a.isoformat(), min(nxt, b).isoformat()))
        a = nxt
    return out


def fetch_range(schema, start, end, parent="NDX.OPT", budget=30.0, out=OUT):
    """One schema over a long window, downloaded a month at a time."""
    paths, spent = [], 0.0
    for a, b in month_edges(start, end):
        cost, size = plan(schema, a, b, parent)
        if spent + cost > budget:
            print(f"  stopping at {a}: ${spent+cost:.2f} would pass the "
                  f"${budget:.2f} budget", flush=True)
            break
        paths.append(fetch(schema, a, b, parent, budget=budget - spent, out=out))
        spent += cost
    print(f"  {schema}: {len(paths)} months, about ${spent:.2f}", flush=True)
    return paths, spent


def fetch(schema, start, end, parent="NDX.OPT", budget=30.0, out=OUT):
    """Download one schema after pricing it against a budget."""
    cost, size = plan(schema, start, end, parent)
    print(f"{schema}: {size/1024**3:.3f} GiB, ${cost:.2f} (budget ${budget:.2f})",
          flush=True)
    if cost > budget:
        raise SystemExit(f"refusing: ${cost:.2f} over the ${budget:.2f} budget")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"{parent.replace('.','_')}_{schema}_{start}_{end}.dbn.zst")
    if os.path.exists(path):
        print(f"  already have {path}", flush=True)
        return path
    c = client()
    data = c.timeseries.get_range(dataset=DATASET, symbols=[parent],
                                  stype_in="parent", schema=schema,
                                  start=start, end=end)
    data.to_file(path)
    print(f"  wrote {path} ({os.path.getsize(path)/1024**2:.1f} MiB)", flush=True)
    return path


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-08-20"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08-20"
    total = 0.0
    for sch, budget in (("definition", 3.0), ("ohlcv-1d", 8.0),
                        ("statistics", 22.0)):
        _, spent = fetch_range(sch, start, end, budget=budget)
        total += spent
    print(f"\nspent about ${total:.2f}")
