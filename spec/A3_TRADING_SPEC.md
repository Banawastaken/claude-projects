# A3 Donchian H4 Swing — executable trading specification

XAUUSD only. Written so a human or an EA produces identical decisions.
Every number here is the one the backtest used; nothing is rounded for
readability.

---

## 1. Data and clocks

| Item | Value |
|---|---|
| Instrument | XAUUSD (FundedNext CFD) |
| Signal timeframe | H4 (four-hour bars) |
| Execution timeframe | M1 or H1 — signals are evaluated on the current price, not on an H4 close |
| Broker day boundary | 00:00 server time (EET/EEST, i.e. 21:00–22:00 UTC) |
| Contract size | 100 oz per 1.00 lot → **$100 per $1.00 of price move per lot** |

All indicator values come from **completed** H4 bars only. The bar currently
forming is never used. This is the single most common way a live EA
accidentally outperforms its backtest.

---

## 2. Indicators (all on completed H4 bars)

| Name | Definition |
|---|---|
| `ATR` | Wilder ATR, period 14, on H4 |
| `ADX` | Wilder ADX, period 14, on H4 |
| `CH_HIGH` | Highest **high** of the previous 20 completed H4 bars, excluding the current forming bar |
| `CH_LOW` | Lowest **low** of the previous 20 completed H4 bars, excluding the current forming bar |

Warm-up: do not trade until at least 40 completed H4 bars exist.

---

## 3. Entry

Evaluate continuously (each M1/H1 tick or bar close). Let `BID` be the current
bid and `ASK` the current ask.

**Long** when all hold:
1. `ADX >= 15`
2. `BID > CH_HIGH`
3. All gates in §6 pass

**Short** when all hold:
1. `ADX >= 15`
2. `BID < CH_LOW`
3. All gates in §6 pass

Order type is market. In the backtest the fill happens on the **next** bar's
open, never the bar that generated the signal — an EA acting on the current
tick is therefore slightly *ahead* of the tested behaviour, which is
acceptable; acting a bar late is not a problem either.

---

## 4. Stop, target, sizing

Let `A = ATR` at entry.

```
SL_DIST = clip(1.0 * A, 0.5 * A, 3.0 * A)      # = 1.0 * A in practice
TP_DIST = 4.0 * SL_DIST                         # 4R
```

Long: `SL = ENTRY - SL_DIST`, `TP = ENTRY + TP_DIST`
Short: `SL = ENTRY + SL_DIST`, `TP = ENTRY - TP_DIST`

where `ENTRY` is the actual fill price (ask for a long, bid for a short).

**Position size**

```
RISK_PCT  = 0.0075                     # 0.75% of balance
if stage is Phase 1 or Phase 2:
    RISK_PCT = 0.0075 * 1.35 = 0.010125
RISK_USD  = BALANCE * RISK_PCT
RAW_LOTS  = RISK_USD / (SL_DIST * 100)
LOTS      = round(RAW_LOTS, 2)         # nearest 0.01
LOTS      = max(LOTS, 0.01)
```

Then apply the caps in §6 and re-check; if the trade is rejected, do not open it.

---

## 5. Trade management

Let `R = SL_DIST` and let `MOVE` be the favourable excursion in price
(`BID - ENTRY` for a long, `ENTRY - ASK` for a short).

| Trigger | Action |
|---|---|
| `MOVE >= 1.2 * R`, once only | Move SL to `ENTRY + 0.15 * R` (long) or `ENTRY - 0.15 * R` (short) |
| `MOVE >= 2.0 * R`, continuously | Trail SL to `BID - 1.5 * R` (long) or `ASK + 1.5 * R` (short) |

The stop only ever moves in the favourable direction — never widen it.
Take-profit is a resting limit at `TP` and is never moved.

There is no time-based exit. Positions may be held over the weekend
(FundedNext permits weekend holding on this account type).

---

## 6. Gates — check every one before opening a trade

Reject the trade unless **all** pass.

| # | Gate | Rule |
|---|---|---|
| 1 | Session | Server hour is within 01:00–21:00 UTC |
| 2 | Weekday | Monday–Friday only |
| 3 | Trade cap | No trade opened yet today (max **1 per day**) |
| 4 | Loss cap | No losing trade closed yet today (max **1 loss per day**) |
| 5 | Daily stop | Today's realised P&L is better than `-2.0%` of the day-start balance |
| 6 | Cool-off | Fewer than 4 consecutive losses; after 4, stand down until the next server day |
| 7 | Spread | Current spread ≤ **1.8 × the instrument's median spread** (≈ $1.10 on gold) |
| 8 | Untradeable size | `0.01 * SL_DIST * 100` ≤ `1.25%` of equity. If the minimum lot alone risks more than that, the account is too small for this setup — **skip the trade** |
| 9 | Room to the limits | `LOTS * SL_DIST * 100` ≤ 80% of the distance to the daily loss limit, and ≤ 80% of the distance to the max loss floor |
| 10 | Funded open risk | In the funded stage, open risk ≤ 3% of equity at all times |

Gate 8 matters more than it looks. When gold's H4 ATR is wide, 0.01 lots can
already risk over 1.25% of a $6,000 account, and taking the trade anyway
over-risks exactly when volatility is highest.

---

## 7. Account rules being traded against

Stellar 2-Step $6,000:

| Rule | Value |
|---|---|
| Phase 1 target | +8% ($6,480) |
| Phase 2 target | +5% ($6,300) |
| Daily loss limit | 5% of day-start balance |
| Max loss | 10% **static** — a hard floor at **$5,400**, it does not trail |
| Minimum trading days | 5 |
| Payout: minimum growth | 2% ($120) |
| Payout: consistency | best single day ≤ 40% of the profit withdrawn |

**Withdrawal policy.** Do not request the first reward at the 21-day minimum.
At three weeks the profit sits in one or two trades and the best day is
typically 76–122% of the total, which fails the 40% consistency check. Waiting
raises both the compliance rate and the amount: median best-day share falls
from 93% at 30 days to 35% at 90 days, and simulated payouts rose from $415 at
a 21-day cadence to $802 at 120 days.

---

## 8. Known behaviour to expect

These come from the simulations and are not guarantees.

- Roughly **1 trade every 3–5 days**. Long quiet stretches are normal; the ADX
  filter stands the strategy down in ranges.
- Win rate around **50–55%**, average winner ~1.6R, average loser ~1.0R.
- Expect a **9–13% peak-to-trough drawdown** at some point. That is inside the
  10% static rule only because the floor is measured from the starting balance,
  not from your peak — do not confuse the two.
- The worst single day in testing was **4.5%** against a 5% limit. One more
  losing trade that day would have ended the account.
- Funded accounts in long simulations produced about **one payout, then died**.
  Withdrawing resets the balance to $6,000 while the floor stays at $5,400, so
  the account never builds a buffer.

---

## 9. Before going live

1. **Confirm the consistency rule applies to your payout path** with FundedNext
   support in writing. It is documented against on-demand reward requests; if
   it does not apply to the standard cycle, withdraw earlier and more often.
2. **Check the broker's real gold spread and commission** against the $0.62
   median and 0.0016% notional used here.
3. **Forward-test on demo for at least 20 trades** and compare fill prices,
   spread and slippage against §8.
4. Confirm the EA's ATR, ADX and channel values match the backtest on the same
   historical bars before trusting a single live signal.
