"""Run every registered strategy and write results.json for the report page.

    SSRN_RAW=/path/to/raw python3 -I run.py out.json
"""

from __future__ import annotations

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import strategies as S

RECENT_START = "2016-10-31"  # last 10 years of the common sample


def sharpe(x: pd.Series) -> float | None:
    x = x.dropna()
    if len(x) < 24 or x.std() == 0:
        return None
    return float(x.mean() / x.std() * np.sqrt(12))


def capm(x: pd.Series, m: pd.Series):
    df = pd.concat([x, m], axis=1, join="inner").dropna()
    if len(df) < 36:
        return None, None, None
    y, X = df.iloc[:, 0].values, np.column_stack([np.ones(len(df)), df.iloc[:, 1].values])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    cov = np.linalg.inv(X.T @ X) * (e @ e) / (len(y) - 2)
    return float(b[0] * 12), float(b[0] / np.sqrt(cov[0, 0])), float(b[1])


def stats(ex: pd.Series, rf: pd.Series, mkt: pd.Series, year: int) -> dict:
    ex = ex.dropna()
    ex = ex[ex.index <= rf.index[-1] + pd.offsets.MonthEnd(2)]
    rfa = rf.reindex(ex.index).ffill().fillna(0)
    tot = ex + rfa
    eq = (1 + tot).cumprod()
    yrs = len(ex) / 12
    dd = eq / eq.cummax() - 1
    yearly = (1 + ex).groupby(ex.index.year).prod() - 1
    full_years = ex.groupby(ex.index.year).size()
    yearly = yearly[full_years == 12]
    sr = sharpe(ex)
    alpha, alpha_t, beta = capm(ex, mkt)
    pre = ex[ex.index.year <= year]
    post = ex[ex.index.year > year]
    # equity curve, log10 growth of $1 with T-bill collateral, sampled monthly
    curve = [[d.strftime("%Y-%m"), round(float(np.log10(v)), 4)] for d, v in eq.items()]
    return dict(
        start=ex.index[0].strftime("%Y-%m"), end=ex.index[-1].strftime("%Y-%m"),
        years=round(yrs, 1),
        ann_excess=float(ex.mean() * 12), vol=float(ex.std() * np.sqrt(12)),
        sharpe=sr, t=None if sr is None else float(sr * np.sqrt(yrs)),
        cagr=float(eq.iloc[-1] ** (1 / yrs) - 1), maxdd=float(dd.min()),
        alpha=alpha, alpha_t=alpha_t, beta=beta,
        pct_up_years=float((yearly > 0).mean()) if len(yearly) else None,
        worst_year=float(yearly.min()) if len(yearly) else None,
        best_year=float(yearly.max()) if len(yearly) else None,
        sharpe_pre=sharpe(pre), sharpe_post=sharpe(post),
        n_post=int(len(post)),
        sharpe_recent=sharpe(ex[ex.index > RECENT_START]),
        yearly={int(k): round(float(v), 4) for k, v in yearly.items()},
        curve=curve,
    )


def main(out_path: str):
    rf = S.rf_m_for(pd.date_range("1926-07-31", "2026-11-30", freq="ME"))
    mkt = S.s_mkt()
    rows, series = [], {}
    for r in S.R:
        try:
            ex = r["fn"]()
            st = stats(ex, rf, mkt, r["year"])
            series[r["id"]] = ex
        except Exception:
            traceback.print_exc()
            continue
        row = {k: v for k, v in r.items() if k != "fn"}
        row.update(st)
        rows.append(row)
        print(f'{r["id"]:10s} {st["start"]}..{st["end"]}  SR {st["sharpe"] or 0:5.2f}  '
              f't {st["t"] or 0:5.1f}  pre {st["sharpe_pre"] or float("nan"):5.2f}  '
              f'post {st["sharpe_post"] or float("nan"):5.2f}  10y {st["sharpe_recent"] or float("nan"):5.2f}  '
              f'CAGR {st["cagr"]:6.1%}  DD {st["maxdd"]:6.1%}')
    # correlations over the common 1990+ window where most series overlap
    df = pd.DataFrame(series)
    corr = df[df.index >= "1990-01-01"].corr(min_periods=60).round(2)
    out = dict(
        generated="2026-11-20",
        rows=rows,
        corr=dict(ids=list(corr.columns), m=[[None if pd.isna(v) else float(v) for v in row] for row in corr.values]),
    )
    with open(out_path, "w") as f:
        json.dump(out, f, allow_nan=False, default=lambda x: None)
    print("wrote", out_path, len(rows), "strategies")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results.json")
