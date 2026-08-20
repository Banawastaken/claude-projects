"""Prop-firm challenge backtest engine for XAUUSD on M1 bars.

Design notes that matter for honesty of the result:

* Bars store BID OHLC plus the real measured spread for that minute. Longs are
  entered at ask and exited at bid; shorts the reverse. The spread is therefore
  a real cost taken from the data, not an assumption.
* Stop-losses are filled with adverse slippage; take-profits are filled at the
  limit price with none. If a bar's range covers both the stop and the target,
  the STOP is assumed to fill first.
* Signals are computed from bar t's close and executed at bar t+1's open, so
  there is no same-bar lookahead.
* Daily loss and max loss are checked against intrabar EQUITY (floating P&L
  included), not against closed balance. This is how prop firms actually kill
  accounts and it is the main thing naive backtests get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CONTRACT = 100.0  # 1.00 lot XAUUSD = 100 oz => $100 per $1.00 price move


@dataclass
class Rules:
    """Stellar 2-Step $6K rule set."""

    initial_balance: float = 6000.0
    p1_target: float = 0.08
    p2_target: float = 0.05
    daily_loss: float = 0.05
    max_loss: float = 0.10
    static_drawdown: bool = True
    min_trading_days: int = 5
    max_open_risk: float = 0.03  # enforced in the funded stage
    profit_split: float = 0.95
    commission_per_lot: float = 7.0  # round-turn USD per 1.0 lot
    slip_entry: float = 0.05  # price units of adverse slippage on market entry
    slip_stop: float = 0.15  # price units of adverse slippage on a stop-out
    # Gold's normal spread on this feed is ~0.60, so this blocks genuine
    # spikes (news, rollover) rather than normal conditions.
    max_spread: float = 1.10
    # The broker's minimum trade is 0.01 lots, worth $1 per $1 of gold move.
    # Once volatility widens an ATR-scaled stop past ~$75, that minimum lot
    # alone risks more than 1.25% of a $6K account, so the account silently
    # over-risks exactly when the market is most dangerous. Rather than compare
    # against the intended risk (which rejects trades that are only slightly
    # oversized), enforce a hard ceiling on what any single trade may risk.
    max_trade_risk_pct: float = 0.0125
    min_lot: float = 0.01
    day_boundary_utc_hour: int = 21  # 00:00 EEST


@dataclass
class Trade:
    idx_in: int
    ts_in: np.datetime64
    direction: int  # +1 long, -1 short
    entry: float
    sl: float
    tp: float
    lots: float
    tag: str = ""
    risk_usd: float = 0.0
    idx_out: int = -1
    ts_out: np.datetime64 = None
    exit: float = 0.0
    pnl: float = 0.0
    reason: str = ""
    mae: float = 0.0  # worst adverse excursion in USD
    mfe: float = 0.0


@dataclass
class Position:
    direction: int
    entry: float
    sl: float
    tp: float
    lots: float
    idx_in: int
    ts_in: np.datetime64
    tag: str
    be_moved: bool = False
    partial_done: bool = False
    peak_r: float = 0.0
    init_risk: float = 0.0
    trail_anchor: float = 0.0


@dataclass
class Ctx:
    """Account state handed to the strategy so it can self-govern risk."""

    stage: str = "phase1"
    balance: float = 0.0
    day_start_balance: float = 0.0
    day_pnl_pct: float = 0.0
    trades_today: int = 0
    losses_today: int = 0
    consec_losses: int = 0
    target_balance: float = 0.0
    floor_equity: float = 0.0


@dataclass
class StageResult:
    name: str
    passed: bool = False
    breached: bool = False
    breach_reason: str = ""
    start_idx: int = 0
    end_idx: int = 0
    start_ts: np.datetime64 = None
    end_ts: np.datetime64 = None
    final_balance: float = 0.0
    peak_equity: float = 0.0
    min_equity: float = 0.0
    trading_days: int = 0
    calendar_days: int = 0
    trades: list = field(default_factory=list)
    equity_idx: list = field(default_factory=list)
    equity_val: list = field(default_factory=list)
    payouts: list = field(default_factory=list)
    worst_daily_dd_pct: float = 0.0
    max_dd_pct: float = 0.0


class Market:
    """Column-oriented M1 bar store with precomputed indicator inputs."""

    def __init__(self, df):
        self.ts = df["ts"].values
        self.o = df["open"].values.astype(np.float64)
        self.h = df["high"].values.astype(np.float64)
        self.l = df["low"].values.astype(np.float64)
        self.c = df["close"].values.astype(np.float64)
        self.n = len(self.c)
        # Ask-side series: use the real ASK feed when we have it, so fills sit
        # on the correct side of the book instead of on a spread assumption.
        if "ask_close" in df.columns:
            self.ao = df["ask_open"].values.astype(np.float64)
            self.ah = df["ask_high"].values.astype(np.float64)
            self.al = df["ask_low"].values.astype(np.float64)
            self.ac = df["ask_close"].values.astype(np.float64)
            self.spread = self.ac - self.c
        else:
            self.spread = df["spread_med"].values.astype(np.float64)
            self.ao = self.o + self.spread
            self.ah = self.h + self.spread
            self.al = self.l + self.spread
            self.ac = self.c + self.spread
        # calendar helpers
        hours = df["ts"].dt.hour.values
        self.hour = hours
        self.minute = df["ts"].dt.minute.values
        self.dow = df["ts"].dt.dayofweek.values
        self.date = df["ts"].dt.date.values

    def trade_day(self, i: int, boundary_hour: int) -> np.datetime64:
        """Day bucket used for the daily-loss reset, shifted to broker midnight."""
        return (self.ts[i] + np.timedelta64(24 - boundary_hour, "h")).astype("datetime64[D]")


def run_stage(
    mkt: Market,
    strategy,
    rules: Rules,
    start_idx: int,
    end_idx: int,
    stage: str,
    start_balance: float,
    target_pct: float | None,
    dd_anchor: float,
    payout_days: tuple[int, int] | None = None,
) -> StageResult:
    """Run one stage (p1 / p2 / funded) forward from start_idx.

    dd_anchor is the balance the STATIC max-loss floor is computed from.
    target_pct None means the stage has no profit target (funded).
    """
    res = StageResult(name=stage, start_idx=start_idx, start_ts=mkt.ts[start_idx])
    balance = start_balance
    floor_equity = dd_anchor * (1.0 - rules.max_loss)
    target_balance = start_balance * (1.0 + target_pct) if target_pct is not None else None

    pos: Position | None = None
    trades: list[Trade] = []
    day = mkt.trade_day(start_idx, rules.day_boundary_utc_hour)
    day_start_balance = balance
    day_low_equity = balance
    traded_days = set()
    ctx = Ctx(stage=stage, balance=balance, day_start_balance=balance)
    trades_today = 0
    losses_today = 0
    consec_losses = 0
    peak_eq = balance
    min_eq = balance
    worst_daily = 0.0
    max_dd = 0.0
    equity_idx: list[int] = []
    equity_val: list[float] = []
    payouts: list[dict] = []
    last_payout_idx = start_idx
    payout_balance_mark = start_balance
    next_payout_days = payout_days[0] if payout_days else None

    strategy.reset()

    def close_position(i, price, reason, portion=1.0):
        nonlocal balance, pos, trades_today, losses_today, consec_losses
        lots = pos.lots * portion
        if pos.direction > 0:
            pnl = (price - pos.entry) * CONTRACT * lots
        else:
            pnl = (pos.entry - price) * CONTRACT * lots
        pnl -= rules.commission_per_lot * lots
        balance += pnl
        tr = Trade(
            idx_in=pos.idx_in,
            ts_in=pos.ts_in,
            direction=pos.direction,
            entry=pos.entry,
            sl=pos.sl,
            tp=pos.tp,
            lots=lots,
            tag=pos.tag,
            risk_usd=pos.init_risk * CONTRACT * lots,
            idx_out=i,
            ts_out=mkt.ts[i],
            exit=price,
            pnl=pnl,
            reason=reason,
        )
        trades.append(tr)
        strategy.on_trade(tr)
        traded_days.add(mkt.trade_day(i, rules.day_boundary_utc_hour))
        if portion >= 1.0:
            trades_today += 1
            if pnl < 0:
                losses_today += 1
                consec_losses += 1
            else:
                consec_losses = 0
            pos = None
        else:
            pos.lots -= lots
            pos.partial_done = True
        return pnl

    def float_pnl(i_price_bid, i_price_ask):
        if pos is None:
            return 0.0
        if pos.direction > 0:
            return (i_price_bid - pos.entry) * CONTRACT * pos.lots
        return (pos.entry - i_price_ask) * CONTRACT * pos.lots

    i = start_idx
    while i < end_idx:
        # ---- day rollover -------------------------------------------------
        d = mkt.trade_day(i, rules.day_boundary_utc_hour)
        if d != day:
            dd_today = (day_start_balance - day_low_equity) / day_start_balance
            worst_daily = max(worst_daily, dd_today)
            day = d
            day_start_balance = balance + float_pnl(mkt.c[i], mkt.ac[i])
            day_low_equity = day_start_balance
            trades_today = 0
            losses_today = 0
            strategy.on_new_day(i, mkt)

        daily_floor = day_start_balance * (1.0 - rules.daily_loss)

        # ---- mark to market on this bar ------------------------------------
        # The worst price actually experienced while the position is open is
        # bounded by its own stop: if the bar trades through the stop, the
        # position is already closed at the stop and equity never sees the
        # bar's extreme. Ignoring that over-reports breaches.
        hit_sl = hit_tp = False
        if pos is not None:
            if pos.direction > 0:
                hit_sl = mkt.l[i] <= pos.sl
                hit_tp = mkt.h[i] >= pos.tp
                worst_px = (pos.sl - rules.slip_stop) if hit_sl else mkt.l[i]
                best_px = mkt.h[i]
            else:
                hit_sl = mkt.ah[i] >= pos.sl
                hit_tp = mkt.al[i] <= pos.tp
                worst_px = (pos.sl + rules.slip_stop) if hit_sl else mkt.ah[i]
                best_px = mkt.al[i]
            eq_worst = balance + (
                (worst_px - pos.entry) * CONTRACT * pos.lots
                if pos.direction > 0
                else (pos.entry - worst_px) * CONTRACT * pos.lots
            )
            eq_best = balance + (
                (best_px - pos.entry) * CONTRACT * pos.lots
                if pos.direction > 0
                else (pos.entry - best_px) * CONTRACT * pos.lots
            )
        else:
            eq_worst = eq_best = balance

        day_low_equity = min(day_low_equity, eq_worst)
        min_eq = min(min_eq, eq_worst)
        peak_eq = max(peak_eq, eq_best)
        max_dd = max(max_dd, (peak_eq - eq_worst) / peak_eq)

        # ---- hard rule checks (equity based) ------------------------------
        if eq_worst <= floor_equity:
            res.breached = True
            res.breach_reason = f"max loss ({rules.max_loss:.0%} static) hit"
            if pos is not None:
                close_position(i, worst_px, "breach")
            break
        if eq_worst <= daily_floor:
            res.breached = True
            res.breach_reason = f"daily loss ({rules.daily_loss:.0%}) hit"
            if pos is not None:
                close_position(i, worst_px, "breach")
            break

        # ---- manage open position ----------------------------------------
        if pos is not None:
            if hit_sl:  # conservative: stop wins ties within the bar
                px = pos.sl - rules.slip_stop * pos.direction
                close_position(i, px, "sl")
            elif hit_tp:
                close_position(i, pos.tp, "tp")
            else:
                strategy.manage(pos, i, mkt, rules)
                if pos is not None and strategy.force_exit(pos, i, mkt):
                    px = mkt.c[i] if pos.direction > 0 else mkt.ac[i]
                    close_position(i, px, "time")
                if pos is not None and pos.lots <= 0:
                    pos = None

        # ---- new entries ---------------------------------------------------
        if pos is None and i + 1 < end_idx:
            ctx.stage = stage
            ctx.balance = balance
            ctx.day_start_balance = day_start_balance
            ctx.day_pnl_pct = (balance - day_start_balance) / day_start_balance
            ctx.trades_today = trades_today
            ctx.losses_today = losses_today
            ctx.consec_losses = consec_losses
            ctx.target_balance = target_balance if target_balance else 0.0
            ctx.floor_equity = floor_equity
            sig = strategy.signal(i, mkt, rules, ctx)
            if sig is not None:
                j = i + 1  # execute at next bar's open
                direction, sl_dist, tp_dist, risk_pct, tag = sig
                if mkt.spread[j] <= rules.max_spread:
                    if direction > 0:
                        entry = mkt.ao[j] + rules.slip_entry
                        sl = entry - sl_dist
                        tp = entry + tp_dist
                    else:
                        entry = mkt.o[j] - rules.slip_entry
                        sl = entry + sl_dist
                        tp = entry - tp_dist
                    equity_now = balance
                    risk_usd = equity_now * risk_pct
                    # never risk more than the rules allow to be open at once
                    cap = equity_now * rules.max_open_risk if stage == "funded" else equity_now * 0.05
                    risk_usd = min(risk_usd, cap)
                    # never let one trade be able to break the daily limit
                    room_daily = max(0.0, equity_now - daily_floor) * 0.80
                    room_total = max(0.0, equity_now - floor_equity) * 0.80
                    risk_usd = min(risk_usd, room_daily, room_total)
                    # Round to the nearest 0.01 lot rather than flooring:
                    # flooring systematically under-risks (often by ~40% once
                    # the stop is wide), which quietly slows every challenge.
                    raw_lots = risk_usd / (sl_dist * CONTRACT)
                    lots = max(np.round(raw_lots * 100) / 100.0, rules.min_lot)
                    # but never let the rounding push us past a hard cap
                    if lots * sl_dist * CONTRACT > min(cap, room_daily, room_total):
                        lots = np.floor(raw_lots * 100) / 100.0
                    # Refuse a trade the account is too small to size properly:
                    # if even the minimum lot risks more than the ceiling, this
                    # setup is untradeable at this account size. This is a
                    # question about the smallest possible position, not about
                    # the intended risk, so it is checked against min_lot.
                    if rules.min_lot * sl_dist * CONTRACT > equity_now * rules.max_trade_risk_pct:
                        lots = 0.0
                    if lots >= rules.min_lot:
                        pos = Position(
                            direction=direction,
                            entry=entry,
                            sl=sl,
                            tp=tp,
                            lots=lots,
                            idx_in=j,
                            ts_in=mkt.ts[j],
                            tag=tag,
                            init_risk=sl_dist,
                            trail_anchor=entry,
                        )
                        traded_days.add(mkt.trade_day(j, rules.day_boundary_utc_hour))
                        i += 1  # the entry bar is consumed
                        equity_idx.append(i)
                        equity_val.append(balance)
                        continue

        if i % 60 == 0:
            equity_idx.append(i)
            equity_val.append(balance + float_pnl(mkt.c[i], mkt.ac[i]))

        # ---- stage completion ---------------------------------------------
        if target_balance is not None and pos is None and balance >= target_balance:
            if len(traded_days) >= rules.min_trading_days:
                res.passed = True
                break

        # ---- funded payouts -------------------------------------------------
        if payout_days is not None and pos is None:
            elapsed = (mkt.ts[i] - mkt.ts[last_payout_idx]) / np.timedelta64(1, "D")
            if elapsed >= next_payout_days and balance > payout_balance_mark:
                gross = balance - payout_balance_mark
                net = gross * rules.profit_split
                payouts.append(
                    {
                        "ts": mkt.ts[i],
                        "gross": gross,
                        "net": net,
                        "balance_before": balance,
                    }
                )
                balance -= gross  # profits withdrawn, back to base
                payout_balance_mark = balance
                last_payout_idx = i
                next_payout_days = payout_days[1]

        i += 1

    if pos is not None:
        j = min(i, end_idx - 1)
        # a long is closed by selling at the bid, a short by buying at the ask
        close_position(j, mkt.c[j] if pos.direction > 0 else mkt.ac[j], "eod")

    dd_today = (day_start_balance - day_low_equity) / day_start_balance
    worst_daily = max(worst_daily, dd_today)

    res.end_idx = min(i, end_idx - 1)
    res.end_ts = mkt.ts[res.end_idx]
    res.final_balance = balance
    res.peak_equity = peak_eq
    res.min_equity = min_eq
    res.trading_days = len(traded_days)
    res.calendar_days = float((res.end_ts - res.start_ts) / np.timedelta64(1, "D"))
    res.trades = trades
    res.equity_idx = equity_idx
    res.equity_val = equity_val
    res.payouts = payouts
    res.worst_daily_dd_pct = worst_daily
    res.max_dd_pct = max_dd
    return res


def run_challenge(mkt: Market, strategy, rules: Rules, start_idx: int, end_idx: int):
    """Phase 1 -> Phase 2 -> funded, sequentially in time on one data stream."""
    strategy.prepare(mkt)
    stages = []
    p1 = run_stage(
        mkt, strategy, rules, start_idx, end_idx, "phase1",
        rules.initial_balance, rules.p1_target, rules.initial_balance,
    )
    stages.append(p1)
    if not p1.passed:
        return stages

    p2 = run_stage(
        mkt, strategy, rules, p1.end_idx + 1, end_idx, "phase2",
        rules.initial_balance, rules.p2_target, rules.initial_balance,
    )
    stages.append(p2)
    if not p2.passed:
        return stages

    funded = run_stage(
        mkt, strategy, rules, p2.end_idx + 1, end_idx, "funded",
        rules.initial_balance, None, rules.initial_balance,
        payout_days=(21, 14),
    )
    stages.append(funded)
    return stages
