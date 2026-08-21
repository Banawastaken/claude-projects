"""Run a strategy as a multi-instrument portfolio on one prop account."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Rules  # noqa: E402
from portfolio import run_portfolio  # noqa: E402
from run_instruments import rules_for  # noqa: E402
from universe import BY_FN  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DECADE = os.path.join(HERE, "..", "data", "decade")


def available(names=None):
    out = []
    for f in sorted(os.listdir(DECADE)):
        if not f.endswith(".parquet"):
            continue
        name = f[:-8]
        if names and name not in names:
            continue
        if name not in BY_FN:
            continue
        df = pd.read_parquet(os.path.join(DECADE, f)).sort_values("ts").reset_index(drop=True)
        if len(df) < 5000:
            continue
        out.append((name, df))
    return out


def make(cls, params):
    return type(cls.__name__ + "_v", (cls,), {**params, "name": cls.name})


def build_books(instruments, concepts):
    """One book per (instrument, concept).

    Books are keyed by both, so several concepts can trade the same instrument
    side by side on one account -- which is the point: a small edge only becomes
    useful when it fires often enough, and firing more often means more markets
    and more independent signals rather than more size.
    """
    books_data, strategies, rules_by = [], {}, {}
    for name, df in instruments:
        for cls, params in concepts:
            key = f"{name}|{cls.__name__[:12]}"
            books_data.append((key, df))
            strategies[key] = make(cls, params)()
            rules_by[key] = rules_for(BY_FN[name], df)
    return books_data, strategies, rules_by


def continuous(instruments, concepts, start, end, risk=0.0035, max_conc=6):
    """Run with no profit target: measures the edge, not the challenge."""
    books_data, strategies, rules_by = build_books(instruments, concepts)
    port = Rules()
    # limits off so an early breach cannot mask the underlying edge
    port.daily_loss = 0.99
    port.max_loss = 0.99
    return run_portfolio(books_data, strategies, rules_by, port,
                         np.datetime64(start), np.datetime64(end),
                         risk_per_trade=risk, max_concurrent=max_conc)


def challenge(instruments, concepts, start, end, risk=0.0035, max_conc=6):
    """Phase 1 then Phase 2 on the shared account."""
    port = Rules()
    books_data, s1, rules_by = build_books(instruments, concepts)
    p1 = run_portfolio(books_data, s1, rules_by, port,
                       np.datetime64(start), np.datetime64(end),
                       risk_per_trade=risk, max_concurrent=max_conc,
                       target_pct=port.p1_target, start_balance=port.initial_balance)
    if p1 is None or not p1.passed:
        return [p1] if p1 else []
    _, s2, _ = build_books(instruments, concepts)
    p2 = run_portfolio(books_data, s2, rules_by, port,
                       p1.end_ts + np.timedelta64(1, "h"), np.datetime64(end),
                       risk_per_trade=risk, max_concurrent=max_conc,
                       target_pct=port.p2_target, start_balance=port.initial_balance)
    return [p1, p2] if p2 else [p1]


def summarise(r, label):
    if r is None:
        print(f"  {label}: no result")
        return
    n = len(r.trades)
    pnl = np.array([t.pnl for t in r.trades]) if n else np.array([0.0])
    risk = np.array([t.risk_usd for t in r.trades if t.risk_usd > 0])
    rmult = np.array([t.pnl / t.risk_usd for t in r.trades if t.risk_usd > 0])
    wr = (pnl > 0).mean() * 100 if n else 0
    print(f"  {label}: trades {n}  WR {wr:.1f}%  expR {rmult.mean() if len(rmult) else 0:+.3f}  "
          f"totR {rmult.sum() if len(rmult) else 0:+.1f}  "
          f"final ${r.final_balance:,.0f}  maxDD {r.max_dd_pct * 100:.2f}%  "
          f"worstDay {r.worst_daily_dd_pct * 100:.2f}%  days {r.calendar_days:.0f}")


if __name__ == "__main__":
    import strategies_v6 as V6

    # tuned on the design window, mid-plateau rather than best cell
    STR = {"pull_bars": 2, "tp_r": 2.5, "atr_mult": 2.5, "trend_len": 200}
    TSM = {"lookback_days": 60, "atr_mult": 2.0, "tp_r": 4.0}
    VOL = {"squeeze_ratio": 0.6, "atr_mult": 1.5, "tp_r": 3.0}

    instruments = available()
    print(f"{len(instruments)} instruments: {', '.join(n for n, _ in instruments)}\n")

    sets = {
        "reversal only": [(V6.ShortTermReversal, STR)],
        "reversal + momentum": [(V6.ShortTermReversal, STR), (V6.TSMomentum, TSM)],
        "all three": [(V6.ShortTermReversal, STR), (V6.TSMomentum, TSM),
                      (V6.VolContraction, VOL)],
    }

    print("DESIGN WINDOW 2015-2020, continuous, drawdown limits off\n")
    for label, concepts in sets.items():
        for risk, conc in ((0.0035, 6), (0.005, 8)):
            r = continuous(instruments, concepts, "2015-01-01", "2021-01-01",
                           risk=risk, max_conc=conc)
            summarise(r, f"{label:20s} risk {risk:.2%} max {conc}")
        print()
