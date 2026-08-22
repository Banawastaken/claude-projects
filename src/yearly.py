"""Year-by-year view of a set of return streams.

A CAGR over eleven years can be carried by two of them. This is the table that
shows whether it was, and it is the one worth reading before any Sharpe ratio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def yearly_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Compounded return per calendar year, per column."""
    return frame.groupby(frame.index.year).apply(
        lambda g: (1 + g).prod() - 1)


def yearly_table(frame: pd.DataFrame, combined: pd.Series | None = None) -> str:
    y = yearly_returns(frame) * 100
    if combined is not None:
        y = y.copy()
        y["COMBINED"] = (combined.groupby(combined.index.year)
                         .apply(lambda g: (1 + g).prod() - 1) * 100)
    cols = list(y.columns)
    hdr = f"{'year':<7s}" + "".join(f"{c[:13]:>15s}" for c in cols)
    out = [hdr, "-" * len(hdr)]
    for yr, row in y.iterrows():
        out.append(f"{yr:<7d}" + "".join(
            f"{v:>14.2f}%" if np.isfinite(v) else f"{'-':>15s}"
            for v in row.values))
    out.append("-" * len(hdr))
    out.append(f"{'mean':<7s}" + "".join(f"{y[c].mean():>14.2f}%" for c in cols))
    out.append(f"{'median':<7s}" + "".join(f"{y[c].median():>14.2f}%" for c in cols))
    out.append(f"{'worst':<7s}" + "".join(f"{y[c].min():>14.2f}%" for c in cols))
    out.append(f"{'pos yrs':<7s}" + "".join(
        f"{int((y[c] > 0).sum()):>10d}/{int(y[c].notna().sum()):<4d}" for c in cols))
    return "\n".join(out)
