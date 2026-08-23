"""Consensus EPS and announcement timing from Alpha Vantage.

This is the piece the concordance rule needs and nothing free had: for every
quarter, the announcement date, reported EPS, the analyst estimate it was
measured against, and whether the release was pre- or post-market -- which is
exactly the BMO/AMC distinction that decides which session reacts.

The free tier allows 25 requests a day, so every response is cached on disk and
never re-requested. A cache hit costs nothing; a miss costs one of the day's
twenty-five.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

CACHE = "data/av"
URL = "https://www.alphavantage.co/query?function=EARNINGS&symbol={s}&apikey={k}"


_last = [0.0]


def fetch(symbol, key=None, cache=CACHE, gap=20.0, tries=3, backoff=75.0):
    """One symbol, cached forever, paced hard.

    The free tier throttles on two axes -- a burst limit of about one request a
    second and a daily cap of twenty-five -- and answers both with a 200 and a
    prose message rather than an error code. Pacing has to happen before every
    request, not after the successful ones: pausing only on success meant a run
    of throttled replies went out back to back and made the throttling worse.
    """
    key = key or os.environ.get("ALPHAVANTAGE_KEY")
    os.makedirs(cache, exist_ok=True)
    path = os.path.join(cache, f"{symbol}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    if not key:
        raise SystemExit("set ALPHAVANTAGE_KEY")

    for attempt in range(tries):
        wait = gap - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(URL.format(s=symbol, k=key),
                                     headers={"User-Agent": "research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode())
        except Exception as e:
            _last[0] = time.time()
            print(f"  {symbol}: {str(e)[:80]}", flush=True)
            continue
        _last[0] = time.time()
        if "quarterlyEarnings" in d:
            with open(path, "w") as fh:
                json.dump(d, fh)
            return d
        note = (d.get("Information") or d.get("Note") or str(d))
        if "per day" in note:
            # The daily cap: no amount of waiting helps before midnight UTC.
            print(f"  {symbol}: daily cap reached", flush=True)
            return None
        print(f"  {symbol}: throttled, waiting {backoff:.0f}s", flush=True)
        time.sleep(backoff)
    return None


def quarters(symbol, since="2014-01-01", **kw):
    """Announcement rows with a usable estimate, oldest first."""
    d = fetch(symbol, **kw)
    if not d:
        return []
    out = []
    for q in d.get("quarterlyEarnings", []):
        rd = q.get("reportedDate")
        if not rd or rd < since:
            continue
        try:
            rep = float(q["reportedEPS"])
            est = float(q["estimatedEPS"])
        except (TypeError, ValueError, KeyError):
            continue
        out.append({
            "date": rd,
            "reported": rep, "estimate": est,
            "surprise": rep - est,
            # Scale the surprise by the estimate so names with different EPS
            # levels are comparable; guard the near-zero denominators that make
            # a percentage explode.
            "surprise_pct": (rep - est) / abs(est) if abs(est) > 0.05 else None,
            "before_open": (q.get("reportTime") or "").lower().startswith("pre"),
        })
    return sorted(out, key=lambda r: r["date"])


def main(tickers):
    got, missing = {}, []
    for t in tickers:
        rows = quarters(t)
        if rows:
            got[t] = rows
            print(f"  {t:<7s} {len(rows):>3d} quarters  "
                  f"{rows[0]['date']} .. {rows[-1]['date']}  "
                  f"{sum(r['before_open'] for r in rows)} pre-market", flush=True)
        else:
            missing.append(t)
    with open("data/pead/av_earnings.json", "w") as fh:
        json.dump(got, fh)
    print(f"\n{len(got)} names with consensus history; {len(missing)} missing")
    return got


if __name__ == "__main__":
    with open("data/pead/basket20.json") as fh:
        main(json.load(fh))
