"""THE test run: Dec 2025 -> present, on data never used for strategy design.

Produces per-strategy stage results, equity curves, funded-account curves and a
JSON summary. Nothing in here tunes anything; every parameter was fixed on the
2025 development window before this was first executed.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market, Rules  # noqa: E402
from report import (correlation_table, journey_frame, plot_funded_only,  # noqa: E402
                    plot_journey, plot_portfolio)
from run import load, slice_period, trade_stats  # noqa: E402

import strategies_final as F  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")

TEST_START = "2025-12-01"
TEST_END = None  # to the end of the data


def stage_row(name, st, rules):
    s = trade_stats(st.trades)
    return {
        "strategy": name,
        "stage": st.name,
        "outcome": "PASS" if st.passed else ("BREACH" if st.breached else "in progress"),
        "detail": st.breach_reason,
        "start": str(pd.Timestamp(st.start_ts).date()),
        "end": str(pd.Timestamp(st.end_ts).date()),
        "calendar_days": round(st.calendar_days, 1),
        "trading_days": st.trading_days,
        "trades": s.get("trades", 0),
        "win_rate_pct": round(s.get("win_rate", 0) * 100, 1),
        "profit_factor": round(s.get("profit_factor", 0), 2),
        "net_usd": round(st.final_balance - rules.initial_balance, 2),
        "final_balance": round(st.final_balance, 2),
        "max_dd_pct": round(st.max_dd_pct * 100, 2),
        "worst_day_pct": round(st.worst_daily_dd_pct * 100, 2),
        "payouts": len(st.payouts),
        "payout_net_usd": round(sum(p["net"] for p in st.payouts), 2),
    }


def run_until_funded(mkt, cls, rules, i0, i1, max_attempts=6):
    """Run the challenge, restarting after a breach the way a real trader would.

    The firm marks this account type "Reset Applicable: Yes", so a blown
    challenge is a fee, not the end. This returns every attempt, so the report
    can state both whether the strategy got funded and what it cost to get there.
    """
    from engine import run_challenge

    attempts = []
    start = i0
    while start < i1 and len(attempts) < max_attempts:
        strat = cls()
        stages = run_challenge(mkt, strat, rules, start, i1)
        attempts.append(stages)
        if len(stages) > 2:  # reached the funded stage
            break
        last = stages[-1]
        if not last.breached:
            break  # ran out of data rather than failing
        # a reset starts the next day
        nxt = last.end_idx + 1
        while nxt < i1 and mkt.ts[nxt].astype("datetime64[D]") == \
                mkt.ts[last.end_idx].astype("datetime64[D]"):
            nxt += 1
        start = nxt
    return attempts


def main():
    from engine import run_challenge

    rules = Rules()
    df = load("xauusd_m1_clean")
    mkt = Market(df)
    i0, i1 = slice_period(df, TEST_START, TEST_END)
    print(f"TEST WINDOW: {df['ts'].iloc[i0]} -> {df['ts'].iloc[i1 - 1]}  "
          f"({(i1 - i0) / 1440:.0f} trading days of M1 bars)\n")

    rows = []
    results = []
    summary = {}
    os.makedirs(REPORTS, exist_ok=True)
    # Clear charts from previous runs. A stale funded-account chart for a
    # strategy that no longer reaches the funded stage is worse than no chart.
    for stale in os.listdir(REPORTS):
        if stale.endswith(".png"):
            os.remove(os.path.join(REPORTS, stale))

    for cls in F.FINAL:
        attempts = run_until_funded(mkt, cls, rules, i0, i1)
        stages = attempts[-1]  # the attempt that got furthest
        results.append((cls.name, stages))
        print(f"=== {cls.name}  (risk {cls.risk_pct:.2%}/trade)")
        if len(attempts) > 1:
            print(f"   note: {len(attempts) - 1} reset(s) after a failed attempt "
                  f"(fee ${59.99 * (len(attempts) - 1):,.2f}); "
                  f"stages below are the final attempt")
            for a_i, a in enumerate(attempts[:-1]):
                bad = a[-1]
                print(f"     attempt {a_i + 1}: {bad.name} breached "
                      f"({bad.breach_reason}) after {bad.calendar_days:.0f} days")
        for st in stages:
            r = stage_row(cls.name, st, rules)
            rows.append(r)
            print(f"   {st.name:7s} {r['outcome']:11s} {r['detail']:28s} "
                  f"bal ${r['final_balance']:>8,.2f}  trades {r['trades']:>3d}  "
                  f"WR {r['win_rate_pct']:>4.1f}%  PF {r['profit_factor']:>5.2f}  "
                  f"days {r['calendar_days']:>5.1f}  maxDD {r['max_dd_pct']:>5.2f}%  "
                  f"worstDay {r['worst_day_pct']:>5.2f}%")
            if st.payouts:
                print(f"           payouts: {len(st.payouts)} totalling "
                      f"${sum(p['net'] for p in st.payouts):,.2f} net to trader")

        got_funded = len(stages) > 2
        summary[cls.name] = {
            "reached_funded": got_funded,
            "risk_pct": cls.risk_pct,
            "attempts": len(attempts),
            "reset_fees_usd": round(59.99 * (len(attempts) - 1), 2),
            "stages": [stage_row(cls.name, s, rules) for s in stages],
        }
        p = plot_journey(cls.name, stages, mkt, rules,
                         os.path.join(REPORTS, f"{cls.__name__}_journey.png"))
        if got_funded:
            plot_funded_only(cls.name, stages, mkt, rules,
                             os.path.join(REPORTS, f"{cls.__name__}_funded.png"))
        print()

    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(REPORTS, "test_stage_results.csv"), index=False)

    plot_portfolio(results, mkt, rules, os.path.join(REPORTS, "portfolio.png"))

    corr = correlation_table(results)
    if not corr.empty:
        print("Daily P&L correlation between strategies:")
        print(corr.round(2).to_string())
        corr.to_csv(os.path.join(REPORTS, "correlation.csv"))
        iu = np.triu_indices_from(corr.values, k=1)
        summary["_avg_pairwise_correlation"] = float(np.mean(corr.values[iu]))
        print(f"\naverage pairwise correlation: {summary['_avg_pairwise_correlation']:.2f}")

    with open(os.path.join(REPORTS, "test_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"\nWrote reports to {os.path.abspath(REPORTS)}")


if __name__ == "__main__":
    main()
