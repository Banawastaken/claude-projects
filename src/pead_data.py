"""Assemble the PEAD dataset: announcement dates from EDGAR, prices from Yahoo.

Universe selection is the part that decides whether the result means anything.

Any universe picked today carries survivorship bias -- a company that delisted
in 2018 has no current price history, so it cannot be in the sample no matter
how it is chosen. That cannot be fixed with free data, and it is stated rather
than hidden. Two things limit the damage:

  * the tickers are a seeded random sample of everything EDGAR lists, not names
    chosen by me. Whatever bias remains is the survivorship of the data source,
    not my hindsight about which companies did well.
  * PEAD is traded here as a cross-sectional long/short. A bias that lifts every
    surviving name lifts both legs and largely cancels, which it would not for a
    long-only strategy.

Average dollar volume is recorded for every name, because PEAD is strongest in
exactly the small illiquid stocks where trading costs eat it, and that trade-off
has to be measurable rather than assumed.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edgar import company_tickers, earnings_dates, submissions  # noqa: E402
from fetch_yahoo import fetch  # noqa: E402

OUT = "data/pead"
MIN_EARNINGS = 20      # ~5 years of quarters
MIN_DAYS = 1200        # ~5 years of prices


def pick_candidates(n=800, seed=42):
    """A seeded random sample of EDGAR tickers, so the choice is not mine."""
    t = company_tickers()
    names = sorted(k for k in t if k.isalpha() and 1 <= len(k) <= 5)
    rng = random.Random(seed)
    return [(k, *t[k]) for k in rng.sample(names, min(n, len(names)))]


def build_events(candidates, out=OUT):
    """EDGAR pass: cheap, so it runs first and filters before Yahoo is touched."""
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "events.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    keep = {}
    for i, (tk, cik, name) in enumerate(candidates):
        sub = submissions(cik)
        if not sub:
            continue
        ed = earnings_dates(sub)
        if len(ed) >= MIN_EARNINGS:
            keep[tk] = {"cik": cik, "name": name, "events": ed}
        if (i + 1) % 100 == 0:
            print(f"  edgar {i+1}/{len(candidates)}  kept {len(keep)}", flush=True)
    with open(path, "w") as fh:
        json.dump(keep, fh)
    return keep


def build_prices(tickers, out=OUT, gap=1.2):
    """Yahoo pass: slow, so it only runs on names EDGAR already qualified."""
    d = os.path.join(out, "px")
    os.makedirs(d, exist_ok=True)
    ok, bad = [], []
    for i, tk in enumerate(tickers):
        f = os.path.join(d, f"{tk}.parquet")
        if os.path.exists(f):
            ok.append(tk)
            continue
        time.sleep(gap)
        try:
            df = fetch(tk, start="2013-06-01")
        except Exception:
            df = None
        if df is None or len(df) < MIN_DAYS:
            bad.append(tk)
        else:
            df.to_parquet(f, index=False)
            ok.append(tk)
        if (i + 1) % 25 == 0:
            print(f"  yahoo {i+1}/{len(tickers)}  ok {len(ok)}  missing {len(bad)}",
                  flush=True)
    return ok, bad


def main(n=800):
    cands = pick_candidates(n)
    print(f"{len(cands)} candidate tickers sampled from EDGAR (seed 42)")
    events = build_events(cands)
    print(f"{len(events)} have >= {MIN_EARNINGS} earnings 8-Ks since 2014")
    ok, bad = build_prices(sorted(events))
    print(f"{len(ok)} have >= {MIN_DAYS} price days; {len(bad)} unavailable")
    with open(os.path.join(OUT, "universe.json"), "w") as fh:
        json.dump({"ok": ok, "missing": bad, "sampled": n, "seed": 42}, fh)
    print(f"\nuniverse written: {len(ok)} names")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 800)
