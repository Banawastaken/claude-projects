# Research protocol — written before testing anything

The previous strategy failed for one reason: it was designed on 2025 data and
tested on 2025-26 data, so both windows shared the same regime. It looked
excellent in the only two years where its edge existed and lost money in the
other ten. Everything below exists to make that specific mistake impossible to
repeat.

## Windows

| Window | Period | Use |
|---|---|---|
| **Design** | 2015-01 → 2020-12 (6 years) | All idea generation and every parameter choice |
| **Held-out** | 2021-01 → 2026-07 (5.6 years) | Looked at ONCE, at the end, for the final candidate only |

Nothing is chosen, tuned, filtered or discarded on held-out evidence. If a
candidate fails the held-out window, it is reported as a failure — it does not
get adjusted and re-run.

## Acceptance criteria — fixed in advance

A candidate is only worth taking to the held-out window if, on the design
window, it clears all of:

1. **Breadth over time** — positive expectancy in at least **4 of the 6**
   design years. A strategy that makes all its money in one year is the failure
   we are trying to avoid.
2. **Enough trades** — at least **40 trades per year**. Below that, a $6K
   account cannot reach +8% then +5% in a sensible time, and the statistics are
   noise anyway.
3. **Survivable** — breach rate below **10%** in challenge simulations, and a
   median worst day under **3.5%** against the 5% daily limit.
4. **Real after costs** — expectancy above **+0.10 R** per trade with spread,
   commission and slippage charged from the feed.

Then, to be recommended:

5. **Held-out positive** — positive expectancy over 2021-2026 as a whole, and
   positive in at least half of those years.
6. **Breadth across instruments** — positive on at least 3 of the tested CFDs,
   not gold alone. A one-market edge is what we already found and could not
   trust.

## Known biases in the measurements

- **Hourly bars flatter trend strategies** by roughly 1.5-2× versus minute bars,
  because a trailing stop ratchets once an hour rather than once a minute.
  Screening runs on hourly data, so every number is an optimistic bound. The
  final candidate is re-checked on minute data for gold before any
  recommendation.
- **Survivorship in the instrument list** — these are the CFDs FundedNext lists
  today. Instruments delisted before 2026 are absent.
- **One data vendor.** Dukascopy's spreads are its own; a different broker
  changes the cost base.

## What "better" has to mean

Not a higher backtest return. The previous strategy already had that. Better
means the edge is visible in years it was never shown, on markets it was never
fitted to, after costs.
