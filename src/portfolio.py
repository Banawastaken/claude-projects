"""Portfolio engine: one prop account, many instruments, shared limits.

The single-instrument engine answers "does this edge exist". This one answers
the question that actually decides a challenge: can a small edge, spread across
enough markets, produce +8% and then +5% without touching a 5% daily or 10%
static loss limit.

The arithmetic is the whole point. A durable edge of +0.077 R over ~42 trades a
year is 3.2 R per instrument per year -- about 2.4% on one market at 0.75%
risk, which cannot pass anything. The same edge across twelve markets is ~38 R
a year, and because the markets do not lose on the same days, the drawdown does
not scale with the return.

Rules enforced on the shared account:
  * daily loss and static max loss measured on total equity including open P&L
  * a cap on how many positions may be open at once
  * a cap on total open risk, which the funded stage sets at 3%
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine import Ctx, Position, Rules, Trade


@dataclass
class Book:
    """One instrument inside the portfolio."""

    name: str
    mkt: object
    strategy: object
    rules: Rules          # per-instrument contract size, commission, min lot
    idx: np.ndarray       # position of each of its bars on the shared timeline
    pos: Position | None = None
    trades_today: int = 0
    losses_today: int = 0
    consec_losses: int = 0


@dataclass
class PortfolioResult:
    passed: bool = False
    breached: bool = False
    breach_reason: str = ""
    final_balance: float = 0.0
    start_ts: object = None
    end_ts: object = None
    calendar_days: float = 0.0
    trading_days: int = 0
    trades: list = field(default_factory=list)
    equity_ts: list = field(default_factory=list)
    equity_val: list = field(default_factory=list)
    max_dd_pct: float = 0.0
    worst_daily_dd_pct: float = 0.0
    payouts: list = field(default_factory=list)


def build_timeline(books_data):
    """Shared hourly timeline, plus each instrument's index onto it."""
    all_ts = None
    for _, df in books_data:
        t = df["ts"].values.astype("datetime64[h]")
        all_ts = t if all_ts is None else np.union1d(all_ts, t)
    return all_ts


def run_portfolio(books_data, strategies, rules_by_inst, port_rules,
                  start_ts, end_ts, risk_per_trade=0.0035,
                  max_concurrent=6, max_total_risk=0.03,
                  target_pct=None, start_balance=None):
    """Step a shared account through time across every instrument.

    books_data: list of (name, dataframe)
    strategies: dict name -> strategy instance (already constructed)
    """
    from engine import Market

    timeline = build_timeline(books_data)
    m = (timeline >= np.datetime64(start_ts, "h")) & (timeline < np.datetime64(end_ts, "h"))
    timeline = timeline[m]
    if len(timeline) < 100:
        return None

    books = []
    for name, df in books_data:
        mkt = Market(df)
        strat = strategies[name]
        strat.prepare(mkt)
        strat.reset()
        t = df["ts"].values.astype("datetime64[h]")
        # map this instrument's bars onto the shared timeline
        pos_in_tl = np.searchsorted(timeline, t)
        valid = (pos_in_tl < len(timeline)) & np.isin(t, timeline)
        lookup = np.full(len(timeline), -1, dtype=np.int64)
        lookup[pos_in_tl[valid]] = np.flatnonzero(valid)
        books.append(Book(name=name, mkt=mkt, strategy=strat,
                          rules=rules_by_inst[name], idx=lookup))

    balance = start_balance if start_balance is not None else port_rules.initial_balance
    floor_equity = port_rules.initial_balance * (1.0 - port_rules.max_loss)
    target_balance = (start_balance * (1.0 + target_pct)) if target_pct else None

    res = PortfolioResult(start_ts=timeline[0])
    day = None
    day_start_equity = balance
    day_low_equity = balance
    peak_eq = balance
    traded_days = set()
    worst_daily = 0.0
    max_dd = 0.0

    def open_risk(bk):
        if bk.pos is None:
            return 0.0
        return bk.pos.init_risk * bk.rules.contract_size * bk.pos.lots

    def float_pnl(bk, k):
        if bk.pos is None:
            return 0.0
        p = bk.pos
        if p.direction > 0:
            return (bk.mkt.c[k] - p.entry) * bk.rules.contract_size * p.lots
        return (p.entry - bk.mkt.ac[k]) * bk.rules.contract_size * p.lots

    def close(bk, k, price, reason):
        nonlocal balance
        p = bk.pos
        if p.direction > 0:
            pnl = (price - p.entry) * bk.rules.contract_size * p.lots
        else:
            pnl = (p.entry - price) * bk.rules.contract_size * p.lots
        pnl -= bk.rules.commission_per_lot * p.lots
        balance += pnl
        res.trades.append(Trade(
            idx_in=p.idx_in, ts_in=p.ts_in, direction=p.direction, entry=p.entry,
            sl=p.sl, tp=p.tp, lots=p.lots, tag=f"{bk.name}:{p.tag}",
            risk_usd=p.init_risk * bk.rules.contract_size * p.lots,
            idx_out=k, ts_out=bk.mkt.ts[k], exit=price, pnl=pnl, reason=reason))
        bk.strategy.on_trade(res.trades[-1])
        bk.trades_today += 1
        if pnl < 0:
            bk.losses_today += 1
            bk.consec_losses += 1
        else:
            bk.consec_losses = 0
        bk.pos = None
        return pnl

    for ti in range(len(timeline)):
        now = timeline[ti]
        d = (now + np.timedelta64(24 - port_rules.day_boundary_utc_hour, "h")
             ).astype("datetime64[D]")
        if d != day:
            if day is not None:
                worst_daily = max(worst_daily,
                                  (day_start_equity - day_low_equity) / day_start_equity)
            day = d
            eq = balance + sum(float_pnl(b, b.idx[ti]) for b in books if b.idx[ti] >= 0)
            day_start_equity = eq
            day_low_equity = eq
            for b in books:
                b.trades_today = 0
                b.losses_today = 0
                b.strategy.on_new_day(max(b.idx[ti], 0), b.mkt)

        daily_floor = day_start_equity * (1.0 - port_rules.daily_loss)

        # ---- mark to market, worst case across the bar --------------------
        eq_worst = balance
        eq_best = balance
        for b in books:
            k = b.idx[ti]
            if k < 0 or b.pos is None:
                continue
            p = b.pos
            slip = b.rules.slip_stop_spread * b.mkt.spread[k]
            if p.direction > 0:
                hit = b.mkt.l[k] <= p.sl
                worst_px = (p.sl - slip) if hit else b.mkt.l[k]
                best_px = b.mkt.h[k]
                eq_worst += (worst_px - p.entry) * b.rules.contract_size * p.lots
                eq_best += (best_px - p.entry) * b.rules.contract_size * p.lots
            else:
                hit = b.mkt.ah[k] >= p.sl
                worst_px = (p.sl + slip) if hit else b.mkt.ah[k]
                best_px = b.mkt.al[k]
                eq_worst += (p.entry - worst_px) * b.rules.contract_size * p.lots
                eq_best += (p.entry - best_px) * b.rules.contract_size * p.lots

        day_low_equity = min(day_low_equity, eq_worst)
        peak_eq = max(peak_eq, eq_best)
        max_dd = max(max_dd, (peak_eq - eq_worst) / peak_eq)

        if eq_worst <= floor_equity:
            res.breached = True
            res.breach_reason = f"max loss ({port_rules.max_loss:.0%} static) hit"
            break
        if eq_worst <= daily_floor:
            res.breached = True
            res.breach_reason = f"daily loss ({port_rules.daily_loss:.0%}) hit"
            break

        # ---- manage and exit ---------------------------------------------
        for b in books:
            k = b.idx[ti]
            if k < 0 or b.pos is None:
                continue
            p = b.pos
            slip = b.rules.slip_stop_spread * b.mkt.spread[k]
            if p.direction > 0:
                hit_sl = b.mkt.l[k] <= p.sl
                hit_tp = b.mkt.h[k] >= p.tp
            else:
                hit_sl = b.mkt.ah[k] >= p.sl
                hit_tp = b.mkt.al[k] <= p.tp
            if hit_sl:
                close(b, k, p.sl - slip * p.direction, "sl")
            elif hit_tp:
                close(b, k, p.tp, "tp")
            else:
                b.strategy.manage(p, k, b.mkt, b.rules)
                if b.strategy.force_exit(p, k, b.mkt):
                    close(b, k, b.mkt.c[k] if p.direction > 0 else b.mkt.ac[k], "time")

        # ---- entries -------------------------------------------------------
        equity_now = balance + sum(float_pnl(b, b.idx[ti]) for b in books if b.idx[ti] >= 0)
        n_open = sum(1 for b in books if b.pos is not None)
        total_risk = sum(open_risk(b) for b in books)

        for b in books:
            k = b.idx[ti]
            if k < 0 or b.pos is not None or k + 1 >= b.mkt.n:
                continue
            if n_open >= max_concurrent:
                break
            ctx = Ctx(stage="funded" if target_pct is None else "phase1",
                      balance=equity_now, day_start_balance=day_start_equity,
                      day_pnl_pct=(equity_now - day_start_equity) / day_start_equity,
                      trades_today=b.trades_today, losses_today=b.losses_today,
                      consec_losses=b.consec_losses,
                      target_balance=target_balance or 0.0, floor_equity=floor_equity)
            sig = b.strategy.signal(k, b.mkt, b.rules, ctx)
            if sig is None:
                continue
            direction, sl_dist, tp_dist, _risk_pct, tag = sig
            j = k + 1
            if b.mkt.spread[j] > b.rules.max_spread_mult * b.mkt.median_spread:
                continue
            slip = b.rules.slip_entry_spread * b.mkt.spread[j]
            if direction > 0:
                entry = b.mkt.ao[j] + slip
                sl, tp = entry - sl_dist, entry + tp_dist
            else:
                entry = b.mkt.o[j] - slip
                sl, tp = entry + sl_dist, entry - tp_dist

            risk_usd = equity_now * risk_per_trade
            room_daily = max(0.0, equity_now - daily_floor) * 0.80
            room_total = max(0.0, equity_now - floor_equity) * 0.80
            room_open = max(0.0, equity_now * max_total_risk - total_risk)
            risk_usd = min(risk_usd, room_daily, room_total, room_open)
            if risk_usd <= 0:
                continue
            raw_lots = risk_usd / (sl_dist * b.rules.contract_size)
            lots = max(round(raw_lots, 2), b.rules.min_lot)
            if b.rules.min_lot * sl_dist * b.rules.contract_size > \
                    equity_now * b.rules.max_trade_risk_pct:
                continue
            if lots * sl_dist * b.rules.contract_size > min(room_daily, room_total, room_open):
                lots = np.floor(raw_lots * 100) / 100.0
            if lots < b.rules.min_lot:
                continue

            b.pos = Position(direction=direction, entry=entry, sl=sl, tp=tp, lots=lots,
                             idx_in=j, ts_in=b.mkt.ts[j], tag=tag,
                             init_risk=sl_dist, trail_anchor=entry)
            n_open += 1
            total_risk += sl_dist * b.rules.contract_size * lots
            traded_days.add(d)

        if ti % 24 == 0:
            res.equity_ts.append(now)
            res.equity_val.append(equity_now)

        if target_balance is not None and balance >= target_balance and \
                all(b.pos is None for b in books):
            if len(traded_days) >= port_rules.min_trading_days:
                res.passed = True
                break

    for b in books:
        if b.pos is not None:
            k = max(b.idx[min(ti, len(timeline) - 1)], 0)
            close(b, k, b.mkt.c[k] if b.pos.direction > 0 else b.mkt.ac[k], "eod")

    res.final_balance = balance
    res.end_ts = timeline[min(ti, len(timeline) - 1)]
    res.calendar_days = float((res.end_ts - res.start_ts) / np.timedelta64(1, "D"))
    res.trading_days = len(traded_days)
    res.max_dd_pct = max_dd
    res.worst_daily_dd_pct = max(worst_daily,
                                 (day_start_equity - day_low_equity) / day_start_equity)
    return res
