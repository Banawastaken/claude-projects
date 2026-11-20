# SSRN strategy backtests

37 canonical strategies from SSRN papers, rebuilt on public data
(Kenneth French Data Library, AQR Data Library, Yahoo ETFs).

* `data.py` - loaders; set `SSRN_RAW` to the folder holding `ff/`, `aqr/`, `etf/`
* `strategies.py` - one function per strategy plus the paper registry
* `run.py` - computes stats and writes the JSON used by `site/index.html`

```
SSRN_RAW=/path/to/raw python3 -I ssrn/run.py ssrn/site/results.json
```

Library portfolios are gross of costs; rules built here pay 5 bp per unit
ETF turnover and 1 bp per side for daily trades.
