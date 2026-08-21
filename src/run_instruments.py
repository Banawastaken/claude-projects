"""Run the chosen strategy across the FundedNext CFD universe.

A3 (H4 Donchian) was selected on its risk profile rather than its return: it
passed Phase 1 in 100% of simulated challenge starts on both the development
and test windows, never breached an account in either, and had the mildest
worst-day of the four in the test window (1.51% median against a 5% limit).

Each instrument gets its own contract size, commission model and minimum lot,
and its own median spread drives the spike filter and slippage, so nothing is
carried over from gold.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Market, Rules  # noqa: E402
from evaluate import pass_rate_parallel  # noqa: E402
from universe import UNIVERSE, Instrument, commission_per_lot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "instruments")
REPORTS = os.path.join(HERE, "..", "reports")

STRAT_MODULE = "strategies_final"
STRAT_CLASS = "A3_DonchianH4"


def load_instrument(inst: Instrument) -> pd.DataFrame | None:
    path = os.path.join(DATA, f"{inst.fn_name}.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path).sort_values("ts").reset_index(drop=True)


def rules_for(inst: Instrument, df: pd.DataFrame) -> Rules:
    """Per-instrument rule set: contract size, commission and lot floor."""
    med_price = float(df["close"].median())
    return Rules(
        contract_size=inst.usd_per_point,
        commission_per_lot=commission_per_lot(inst, med_price),
        min_lot=inst.min_lot,
    )


def starts_for(df: pd.DataFrame, first: str, last: str, every_days: int = 5) -> list[int]:
    ts = df["ts"].values.astype("datetime64[m]")
    t0 = np.datetime64(pd.Timestamp(first, tz="UTC").tz_localize(None), "m")
    t1 = np.datetime64(pd.Timestamp(last, tz="UTC").tz_localize(None), "m")
    out, cur = [], t0
    while cur < t1:
        idx = int(np.searchsorted(ts, cur))
        if idx < len(ts):
            out.append(idx)
        cur = cur + np.timedelta64(every_days, "D")
    return out


WINDOWS = [
    ("DEV", "2025-02-01", "2025-05-05", 205),
    ("TEST", "2025-12-01", "2026-02-20", 175),
]


def evaluate(inst: Instrument) -> list[dict]:
    df = load_instrument(inst)
    if df is None or len(df) < 3000:
        return []
    rules = rules_for(inst, df)
    mkt = Market(df)
    rows = []
    for label, a, b, horizon in WINDOWS:
        starts = starts_for(df, a, b)
        if len(starts) < 5:
            continue
        r = pass_rate_parallel(
            STRAT_MODULE, STRAT_CLASS, {}, starts, horizon,
            tag=f"INSTRUMENT:{inst.fn_name}",
            rule_over={
                "contract_size": rules.contract_size,
                "commission_per_lot": rules.commission_per_lot,
                "min_lot": rules.min_lot,
            },
            workers=4,
        )
        if not r:
            continue
        rows.append({
            "instrument": inst.fn_name,
            "class": inst.asset_class,
            "window": label,
            "runs": r["runs"],
            "p1_pass": r["p1_pass"] * 100,
            "funded": r["funded"] * 100,
            "alive": r["funded_alive"] * 100,
            "breach": r["breach_rate"] * 100,
            "dd_med": r["dd_med"],
            "dd_p90": r["dd_p90"],
            "worstday_med": r["worstday_med"],
            "worstday_max": r["worstday_max"],
            "trades": r["trades_med"],
            "payout": r["payout_when_funded"],
            "spread_bp": mkt.median_spread / float(df["close"].median()) * 10000,
        })
    return rows


if __name__ == "__main__":
    all_rows = []
    for inst in UNIVERSE:
        rows = evaluate(inst)
        if not rows:
            print(f"  {inst.fn_name:8s} skipped (no data)", flush=True)
            continue
        for r in rows:
            print(f"  {r['instrument']:8s} {r['window']:5s} P1 {r['p1_pass']:3.0f}%  "
                  f"fund {r['funded']:3.0f}%  alive {r['alive']:3.0f}%  "
                  f"breach {r['breach']:3.0f}%  ddMed {r['dd_med']:5.2f}  "
                  f"wdMax {r['worstday_max']:5.2f}  trades {r['trades']:3.0f}  "
                  f"pay ${r['payout']:5.0f}", flush=True)
        all_rows.extend(rows)
    out = pd.DataFrame(all_rows)
    os.makedirs(REPORTS, exist_ok=True)
    out.to_csv(os.path.join(REPORTS, "instrument_results.csv"), index=False)
    print(f"\nwrote {len(out)} rows -> reports/instrument_results.csv")
