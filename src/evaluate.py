"""Evaluation harness: raw edge measurement and challenge pass-rate Monte Carlo.

Two very different questions get asked here:

1. `raw_edge` - does the signal make money at all? Run it with the prop-firm
   limits switched off so a single early breach cannot hide a good edge, and
   report expectancy in R.
2. `pass_rate` - would it survive the actual rules? Start a fresh challenge on
   many different dates and report how often it gets funded. A single
   chronological run is one sample and tells you almost nothing; the
   distribution over start dates is the honest answer.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market, Rules, run_challenge, run_stage  # noqa: E402


def unlimited_rules(base: Rules | None = None) -> Rules:
    r = replace(base or Rules())
    r.daily_loss = 0.99
    r.max_loss = 0.99
    return r


def raw_edge(strategy, mkt: Market, i0: int, i1: int, rules: Rules | None = None) -> dict:
    """Expectancy of the signal with the drawdown rules disabled."""
    rules = unlimited_rules(rules)
    strategy.prepare(mkt)
    st = run_stage(mkt, strategy, rules, i0, i1, "funded", rules.initial_balance, None,
                   rules.initial_balance)
    tr = st.trades
    if not tr:
        return {
            "trades": 0, "win_rate": 0.0, "expectancy_R": 0.0, "total_R": 0.0,
            "avg_win_R": 0.0, "avg_loss_R": 0.0, "pf": 0.0,
            "max_dd_pct": 0.0, "final": rules.initial_balance,
        }
    r_mult = np.array([t.pnl / t.risk_usd if t.risk_usd > 0 else 0.0 for t in tr])
    wins = r_mult[r_mult > 0]
    losses = r_mult[r_mult <= 0]
    gp, gl = wins.sum(), -losses.sum()
    return {
        "trades": len(tr),
        "win_rate": len(wins) / len(r_mult),
        "expectancy_R": float(r_mult.mean()),
        "total_R": float(r_mult.sum()),
        "avg_win_R": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_R": float(losses.mean()) if len(losses) else 0.0,
        "pf": float(gp / gl) if gl > 0 else float("inf"),
        "max_dd_pct": st.max_dd_pct * 100,
        "final": st.final_balance,
    }


def pass_rate(
    strategy_cls,
    mkt: Market,
    df: pd.DataFrame,
    starts: list[int],
    horizon_days: int,
    rules: Rules | None = None,
) -> dict:
    """Start a fresh challenge at each index in `starts` and tally outcomes."""
    rules = rules or Rules()
    minutes = horizon_days * 1440
    p1 = p2 = fund = alive = 0
    p1_days, p2_days = [], []
    fund_pnl = []
    breach_reasons = {}
    for s in starts:
        e = min(s + minutes, mkt.n)
        if e - s < 20 * 1440:
            continue
        strat = strategy_cls()
        stages = run_challenge(mkt, strat, rules, s, e)
        st1 = stages[0]
        if st1.passed:
            p1 += 1
            p1_days.append(st1.calendar_days)
        elif st1.breached:
            breach_reasons[st1.breach_reason] = breach_reasons.get(st1.breach_reason, 0) + 1
        if len(stages) > 1:
            st2 = stages[1]
            if st2.passed:
                p2 += 1
                p2_days.append(st2.calendar_days)
            elif st2.breached:
                breach_reasons["P2 " + st2.breach_reason] = (
                    breach_reasons.get("P2 " + st2.breach_reason, 0) + 1
                )
        if len(stages) > 2:
            fund += 1
            stf = stages[2]
            if not stf.breached:
                alive += 1
            fund_pnl.append(sum(p["net"] for p in stf.payouts))
    n = len(starts)
    return {
        "runs": n,
        "p1_pass": p1 / n if n else 0,
        "p2_pass": p2 / n if n else 0,
        "funded": fund / n if n else 0,
        "funded_alive": alive / n if n else 0,
        "med_p1_days": float(np.median(p1_days)) if p1_days else float("nan"),
        "med_p2_days": float(np.median(p2_days)) if p2_days else float("nan"),
        "avg_payout": float(np.mean(fund_pnl)) if fund_pnl else 0.0,
        "breaches": breach_reasons,
    }


_CACHE = {}


def _market(tag):
    if tag not in _CACHE:
        from run import load

        df = load(tag)
        _CACHE[tag] = (Market(df), df)
    return _CACHE[tag]


def _one_run(args):
    """One challenge from one start index, run in a worker process."""
    mod, cls_name, params, tag, s, horizon_days, rule_over = args
    import importlib

    m = importlib.import_module(mod)
    base = getattr(m, cls_name)
    cls = type(cls_name + "_v", (base,), dict(params)) if params else base
    mkt, _ = _market(tag)
    rules = Rules(**rule_over) if rule_over else Rules()
    e = min(s + horizon_days * 1440, mkt.n)
    if e - s < 20 * 1440:
        return None
    stages = run_challenge(mkt, cls(), rules, s, e)
    out = {
        "start": s,
        "p1": stages[0].passed,
        "p1_breach": stages[0].breach_reason if stages[0].breached else "",
        "p1_days": stages[0].calendar_days,
        "p2": len(stages) > 1 and stages[1].passed,
        "p2_days": stages[1].calendar_days if len(stages) > 1 else float("nan"),
        "p2_breach": (stages[1].breach_reason if len(stages) > 1 and stages[1].breached else ""),
        "funded": len(stages) > 2,
        "funded_alive": len(stages) > 2 and not stages[2].breached,
        "funded_breach": (stages[2].breach_reason if len(stages) > 2 and stages[2].breached else ""),
        "payout": sum(p["net"] for p in stages[2].payouts) if len(stages) > 2 else 0.0,
        "n_payouts": len(stages[2].payouts) if len(stages) > 2 else 0,
    }
    return out


def pass_rate_parallel(mod, cls_name, params, starts, horizon_days=150,
                       tag="xauusd_m1_clean", rule_over=None, workers=4) -> dict:
    jobs = [(mod, cls_name, params, tag, s, horizon_days, rule_over) for s in starts]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(_one_run, jobs):
            if r is not None:
                rows.append(r)
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    breaches = {}
    for col in ("p1_breach", "p2_breach", "funded_breach"):
        for v in d[col]:
            if v:
                key = col.split("_")[0].upper() + " " + v
                breaches[key] = breaches.get(key, 0) + 1
    return {
        "runs": len(d),
        "p1_pass": d["p1"].mean(),
        "p2_pass": d["p2"].mean(),
        "funded": d["funded"].mean(),
        "funded_alive": d["funded_alive"].mean(),
        "med_p1_days": d.loc[d["p1"], "p1_days"].median(),
        "med_p2_days": d.loc[d["p2"], "p2_days"].median(),
        "avg_payout": d["payout"].mean(),
        "payout_when_funded": d.loc[d["funded"], "payout"].mean() if d["funded"].any() else 0.0,
        "breaches": breaches,
        "detail": d,
    }


def month_starts(df: pd.DataFrame, start: str, end: str, every_days: int = 7) -> list[int]:
    """Indices of the first bar on or after each start date in the window."""
    ts = df["ts"].values
    t0 = np.datetime64(pd.Timestamp(start, tz="UTC").tz_localize(None))
    t1 = np.datetime64(pd.Timestamp(end, tz="UTC").tz_localize(None))
    naive = ts.astype("datetime64[m]")
    out = []
    cur = t0
    while cur < t1:
        idx = np.searchsorted(naive, cur)
        if idx < len(naive):
            out.append(int(idx))
        cur = cur + np.timedelta64(every_days, "D")
    return out
