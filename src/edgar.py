"""Earnings announcement dates from SEC EDGAR.

EDGAR is the primary source: an 8-K carrying item 2.02, "Results of Operations
and Financial Condition", is the filing a company makes when it announces
earnings, and its filing date is the announcement date. That is free, official,
and goes back decades -- which is why PEAD is testable at all here.

SEC's fair-access policy requires a declared contact in the User-Agent and caps
automated traffic at 10 requests a second. Both are honoured below. The contact
is the repository owner's, used with their explicit say-so and sent to SEC
alone.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

CONTACT = os.environ.get("SEC_CONTACT", "Mikhail Hoh mikhailhoh@gmail.com")
HEADERS = {"User-Agent": CONTACT, "Accept-Encoding": "gzip, deflate",
           "Host": "data.sec.gov"}
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUB_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SEC allows 10 requests a second; sit comfortably under it.
_last = [0.0]
MIN_GAP = 0.14


def _get(url, host=None, tries=4):
    delay = 2.0
    for _ in range(tries):
        gap = MIN_GAP - (time.time() - _last[0])
        if gap > 0:
            time.sleep(gap)
        h = dict(HEADERS)
        if host:
            h["Host"] = host
        else:
            h.pop("Host", None)
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                _last[0] = time.time()
                return json.loads(raw.decode())
        except Exception:
            _last[0] = time.time()
            time.sleep(delay)
            delay *= 2
    return None


def company_tickers(cache="data/edgar/company_tickers.json"):
    """Every ticker EDGAR knows, as {ticker: (cik, name)}."""
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        with open(cache) as fh:
            d = json.load(fh)
    else:
        d = _get(TICKERS_URL, host="www.sec.gov")
        if not d:
            return {}
        with open(cache, "w") as fh:
            json.dump(d, fh)
    return {v["ticker"]: (int(v["cik_str"]), v["title"]) for v in d.values()}


def submissions(cik: int, cache_dir="data/edgar/sub"):
    """All filings for a CIK, following EDGAR's older-filing shards.

    The `recent` block holds only the last 1,000 filings; a company that files
    often can burn that in a couple of years, so the shards matter for a
    decade-long study.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{cik:010d}.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)

    d = _get(SUB_URL.format(cik=cik))
    if not d:
        return None
    filings = d.get("filings", {})
    rows = dict(filings.get("recent", {}))
    for extra in filings.get("files", []):
        more = _get(f"https://data.sec.gov/submissions/{extra['name']}")
        if not more:
            continue
        for k, v in more.items():
            rows.setdefault(k, [])
            rows[k] = list(rows[k]) + list(v)
    out = {"cik": cik, "name": d.get("name"), "tickers": d.get("tickers", []),
           "filings": rows}
    with open(cache, "w") as fh:
        json.dump(out, fh)
    return out


def earnings_dates(sub, since="2014-01-01"):
    """Announcement dates: 8-K filings carrying item 2.02.

    `acceptanceDateTime` matters as well as the filing date -- a release
    accepted after 16:00 ET is news for the *next* session, and treating it as
    same-day information would be a look-ahead of exactly one day, which is the
    whole horizon this strategy trades on.
    """
    f = sub.get("filings", {})
    forms = f.get("form", [])
    dates = f.get("filingDate", [])
    items = f.get("items", [])
    accept = f.get("acceptanceDateTime", [])
    out = []
    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        it = items[i] if i < len(items) else ""
        if "2.02" not in (it or ""):
            continue
        d = dates[i]
        if d < since:
            continue
        out.append({"date": d,
                    "accepted": accept[i] if i < len(accept) else ""})
    return sorted(out, key=lambda r: r["date"])


if __name__ == "__main__":
    t = company_tickers()
    print(f"{len(t):,} tickers known to EDGAR")
    for tk in sys.argv[1:] or ["AAPL", "MSFT"]:
        cik, name = t.get(tk, (None, None))
        if cik is None:
            print(f"{tk}: not found")
            continue
        sub = submissions(cik)
        ed = earnings_dates(sub)
        print(f"\n{tk} ({name}, CIK {cik}): {len(ed)} earnings 8-Ks since 2014")
        for r in ed[:3] + ed[-3:]:
            print(f"   {r['date']}  accepted {r['accepted']}")


def shares_outstanding(cik: int, cache_dir="data/edgar/shares"):
    """Split-adjusted common shares outstanding, by filing date.

    Raw XBRL share counts are not split-adjusted: Apple reports 895M shares in
    2009 and 14.6bn now, which is two stock splits and not dilution. Between
    consecutive quarterly filings a real buyback or raise moves the count by a
    few per cent at most, so any jump beyond +-50% is a split and is divided
    out. The series is then rebased forward, which is all net issuance needs.
    """
    import os
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{cik:010d}.json")
    if os.path.exists(cache):
        with open(cache) as fh:
            return json.load(fh)

    rows = []
    for concept in ("dei/EntityCommonStockSharesOutstanding",
                    "us-gaap/CommonStockSharesOutstanding"):
        d = _get(f"https://data.sec.gov/api/xbrl/companyconcept/"
                 f"CIK{cik:010d}/{concept}.json")
        if not d:
            continue
        for unit, obs in (d.get("units") or {}).items():
            for o in obs:
                end = o.get("end") or o.get("filed")
                val = o.get("val")
                if end and val:
                    rows.append({"date": end, "shares": float(val)})
        if rows:
            break

    if not rows:
        with open(cache, "w") as fh:
            json.dump([], fh)
        return []

    by_date = {}
    for r in rows:                      # keep the largest report per date
        d0 = r["date"]
        by_date[d0] = max(by_date.get(d0, 0.0), r["shares"])
    series = [{"date": d, "shares": s} for d, s in sorted(by_date.items())]

    factor = 1.0
    out = []
    for i, r in enumerate(series):
        if i:
            ratio = r["shares"] / series[i - 1]["shares"]
            if ratio > 1.5 or ratio < 0.667:
                factor *= ratio         # a split, not a share issue
        out.append({"date": r["date"], "shares": r["shares"] / factor})
    with open(cache, "w") as fh:
        json.dump(out, fh)
    return out
