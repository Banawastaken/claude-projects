"""Loaders for the public datasets the SSRN strategy backtests run on.

Sources (all free, all downloaded by fetch.sh):
  * Kenneth French Data Library - CRSP-based US factor and sorted-portfolio
    returns, 1926 onward. These ARE the long-short portfolios behind most of
    the classic anomaly papers, built by the people who wrote many of them.
  * AQR Data Library - BAB, QMJ, time-series momentum, value & momentum
    everywhere.
  * Yahoo Finance - adjusted daily ETF prices for the tactical-allocation
    papers, which were written about ETF-implementable portfolios.

Every return series is returned as a decimal (0.01 = 1%), indexed by period end.
"""

from __future__ import annotations

import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

RAW = os.environ.get("SSRN_RAW", os.path.join(os.path.dirname(__file__), "raw"))


# ---------------------------------------------------------------- French ----
def _french_sections(name: str) -> dict[str, pd.DataFrame]:
    """Split a French CSV into its titled sections, keeping only date-keyed rows."""
    with zipfile.ZipFile(os.path.join(RAW, "ff", f"{name}.zip")) as z:
        text = z.read(z.namelist()[0]).decode("latin-1")
    sections: dict[str, pd.DataFrame] = {}
    title, header, rows = "main", None, []

    def flush():
        if header is not None and rows:
            df = pd.DataFrame(rows, columns=["date"] + header)
            sections.setdefault(title, df)

    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(","):  # header row
            flush()
            header, rows = [c.strip() for c in s.split(",")[1:]], []
            continue
        parts = [p.strip() for p in s.split(",")]
        if re.fullmatch(r"\d{6}|\d{8}", parts[0]) and header is not None:
            rows.append([parts[0]] + parts[1 : len(header) + 1])
        elif not re.fullmatch(r"\d{4}", parts[0]):
            flush()
            title, header, rows = s, None, []
    flush()
    out = {}
    for k, df in sections.items():
        d = df["date"]
        if d.str.len().iloc[0] == 6:
            idx = pd.to_datetime(d, format="%Y%m") + pd.offsets.MonthEnd(0)
        else:
            idx = pd.to_datetime(d, format="%Y%m%d")
        v = df.drop(columns="date").apply(pd.to_numeric, errors="coerce")
        v = v.mask(v <= -99.99) / 100.0
        v.index = idx
        out[k] = v
    return out


def french(name: str, section: str | None = None) -> pd.DataFrame:
    """Return one section of a French file (first section by default)."""
    secs = _french_sections(name)
    if section is None:
        return next(iter(secs.values()))
    for k, v in secs.items():
        if section.lower() in k.lower():
            return v
    raise KeyError(f"{section!r} not in {name}: {list(secs)}")


# ------------------------------------------------------------------- AQR ----
def aqr(name: str, sheet: str, column: str, header_row: int | None = None) -> pd.Series:
    """Read one column from an AQR workbook. The header row is found by
    looking for a first cell of 'DATE'."""
    path = os.path.join(RAW, "aqr", f"{name}.xlsx")
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    if header_row is None:
        first = raw.iloc[:, 0].astype(str).str.strip().str.upper()
        header_row = int(np.flatnonzero(first == "DATE")[0]) if (first == "DATE").any() else None
    if header_row is None:  # TSMOM file: header in col B, dates in col A
        for i in range(40):
            if column in [str(x).strip() for x in raw.iloc[i].tolist()]:
                header_row = i
                break
    cols = [str(c).strip() for c in raw.iloc[header_row]]
    df = raw.iloc[header_row + 1 :].copy()
    df.columns = cols
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    s = pd.to_numeric(df[column], errors="coerce")
    s.index = dates + pd.offsets.MonthEnd(0)
    return s[s.index.notna()].dropna()


# ------------------------------------------------------------------- ETFs ---
def etf(tickers: list[str], field: str = "Close") -> pd.DataFrame:
    frames = {}
    for t in tickers:
        d = pd.read_csv(os.path.join(RAW, "etf", f"{t}.csv"), index_col=0, parse_dates=True)
        frames[t] = d[field]
    return pd.DataFrame(frames).sort_index()


def etf_ohlc(ticker: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RAW, "etf", f"{ticker}.csv"), index_col=0, parse_dates=True)


# -------------------------------------------------------------- risk-free --
def rf_daily() -> pd.Series:
    """French daily T-bill rate, extended past its last date with ^IRX."""
    rf = french("F-F_Research_Data_Factors_daily")["RF"]
    try:
        irx = etf(["^IRX"])["^IRX"] / 100 / 252
        tail = irx[irx.index > rf.index[-1]]
        rf = pd.concat([rf, tail])
    except FileNotFoundError:
        pass
    return rf


def rf_monthly() -> pd.Series:
    return french("F-F_Research_Data_Factors")["RF"]
