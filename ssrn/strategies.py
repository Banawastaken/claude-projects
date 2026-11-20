"""SSRN strategy library.

Each strategy is a function returning a MONTHLY EXCESS-RETURN series
(return over the 1-month T-bill). Long-short portfolios are self-financing, so
their raw spread already is an excess return. Long-only and timing strategies
have the T-bill subtracted. Equity curves and CAGRs elsewhere add the T-bill
back, i.e. they assume collateral earns cash.

Signals use information available at the close of period t and are applied
to returns in t+1. Where a strategy is built from positions here (timing,
ETF rotation, calendar rules), trading costs are charged as described in its
`costs` field. Portfolios taken directly from the French or AQR libraries are
GROSS of trading costs, as published.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data as D

ETF_COST = 0.0005  # 5 bp per unit of one-way turnover on liquid ETFs
DAY_COST = 0.0001  # 1 bp per side for daily market in/out on SPY-type vehicle


# ------------------------------------------------------------ helpers ------
def to_monthly_excess(total_daily: pd.Series, rf_daily: pd.Series) -> pd.Series:
    rf = rf_daily.reindex(total_daily.index).ffill().fillna(0)
    tot = (1 + total_daily).resample("ME").prod() - 1
    rfm = (1 + rf).resample("ME").prod() - 1
    return (tot - rfm).dropna()


def ls(df: pd.DataFrame, long: str, short: str) -> pd.Series:
    return (df[long] - df[short]).dropna()


def month_end_prices(tickers):
    px = D.etf(tickers)
    return px.resample("ME").last()


def rf_m_for(index) -> pd.Series:
    rfd = D.rf_daily()
    rfm = (1 + rfd).resample("ME").prod() - 1
    return rfm.reindex(index).ffill()


def weights_backtest(w: pd.DataFrame, rets: pd.DataFrame, cash_rf: pd.Series,
                     cost: float = ETF_COST) -> pd.Series:
    """w: target weights decided at month end t (rows), applied to t+1 returns.
    Unallocated weight earns the T-bill. Returns monthly excess return net of
    turnover cost."""
    w = w.fillna(0)
    held = w.shift(1).dropna(how="all")
    r = rets.reindex(held.index)
    rf = cash_rf.reindex(held.index).fillna(0)
    gross = (held * r).sum(axis=1) + (1 - held.sum(axis=1)) * rf
    turnover = held.diff().abs().sum(axis=1).fillna(held.abs().sum(axis=1))
    net = gross - turnover * cost
    return (net - rf).iloc[1:]


# ---------------------------------------------------------- data caches ----
_c: dict = {}


def ff3():
    if "ff3" not in _c:
        _c["ff3"] = D.french("F-F_Research_Data_Factors")
    return _c["ff3"]


def ff5():
    if "ff5" not in _c:
        _c["ff5"] = D.french("F-F_Research_Data_5_Factors_2x3")
    return _c["ff5"]


def umd():
    return D.french("F-F_Momentum_Factor")["Mom"]


def mkt_total_m():
    f = ff3()
    return f["Mkt-RF"] + f["RF"]


def ffd():
    if "ffd" not in _c:
        _c["ffd"] = D.french("F-F_Research_Data_Factors_daily")
    return _c["ffd"]


# ======================================================= STRATEGIES ========
# --- 1. Cross-sectional equity anomalies (US, CRSP, French library) -------
def s_mkt():
    return ff3()["Mkt-RF"]


def s_smb():
    return ff3()["SMB"]


def s_hml():
    return ff3()["HML"]


def s_umd():
    return umd()


def s_str():
    return D.french("F-F_ST_Reversal_Factor")["ST_Rev"]


def s_ltr():
    return D.french("F-F_LT_Reversal_Factor")["LT_Rev"]


def s_rmw():
    return ff5()["RMW"]


def s_cma():
    return ff5()["CMA"]


def s_lowbeta():
    return ls(D.french("Portfolios_Formed_on_BETA"), "Lo 20", "Hi 20")


def s_lowvar():
    return ls(D.french("Portfolios_Formed_on_VAR"), "Lo 20", "Hi 20")


def s_lowivol():
    return ls(D.french("Portfolios_Formed_on_RESVAR"), "Lo 20", "Hi 20")


def s_accruals():
    return ls(D.french("Portfolios_Formed_on_AC"), "Lo 20", "Hi 20")


def s_netissue():
    return ls(D.french("Portfolios_Formed_on_NI"), "< 0", "Hi 20")


def s_ep():
    return ls(D.french("Portfolios_Formed_on_E-P"), "Hi 20", "Lo 20")


def s_cfp():
    return ls(D.french("Portfolios_Formed_on_CF-P"), "Hi 20", "Lo 20")


def s_dp():
    return ls(D.french("Portfolios_Formed_on_D-P"), "Hi 20", "Lo 20")


def s_valmom():
    return (0.5 * s_hml() + 0.5 * s_umd()).dropna()


# --- 2. AQR library factors ----------------------------------------------
def s_bab():
    return D.aqr("Betting-Against-Beta-Equity-Factors-Monthly", "BAB Factors", "USA")


def s_qmj():
    return D.aqr("Quality-Minus-Junk-Factors-Monthly", "QMJ Factors", "USA")


def s_tsmom_aqr():
    return D.aqr("Time-Series-Momentum-Factors-Monthly", "TSMOM Factors", "TSMOM")


def s_vme():
    v = D.aqr("Value-and-Momentum-Everywhere-Factors-Monthly", "VME Factors", "VAL")
    m = D.aqr("Value-and-Momentum-Everywhere-Factors-Monthly", "VME Factors", "MOM")
    return (0.5 * v + 0.5 * m).dropna()


# --- 3. Industry / factor rotation (built here) ---------------------------
def _ind49():
    if "ind" not in _c:
        _c["ind"] = D.french("49_Industry_Portfolios")
    return _c["ind"]


def _rank_ls(score: pd.DataFrame, rets: pd.DataFrame, n: int) -> pd.Series:
    out = {}
    for t in score.index[:-1]:
        s = score.loc[t].dropna()
        nxt = score.index[score.index.get_loc(t) + 1]
        r = rets.loc[nxt]
        s = s[r[s.index].notna()]
        if len(s) < 2 * n:
            continue
        s = s.sort_values()
        out[nxt] = r[s.index[-n:]].mean() - r[s.index[:n]].mean()
    return pd.Series(out)


def s_indmom():
    r = _ind49()
    score = (1 + r).rolling(6, min_periods=6).apply(np.prod, raw=True) - 1
    return _rank_ls(score, r, 7)


def s_indseason():
    r = _ind49()
    score = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for m in range(1, 13):
        sub = r[r.index.month == m]
        # average same-calendar-month return over the prior 20 years
        avg = sub.shift(1).rolling(20, min_periods=5).mean()  # prior years only
        score.loc[sub.index] = avg.values
    # score at month t is for month t (uses only prior years' same month);
    # shift so that the score for month t+1 is known at the end of month t.
    score = score.shift(-1)
    return _rank_ls(score, r, 7)


def s_factormom():
    f = pd.concat([ff5()[["SMB", "HML", "RMW", "CMA"]], umd().rename("UMD"),
                   s_str().rename("STR"), s_ltr().rename("LTR")], axis=1).dropna()
    past = f.rolling(12).sum()
    sign = np.sign(past).shift(1)
    return (sign * f).mean(axis=1).dropna().iloc[1:]


# --- 4. Risk-managed (volatility-scaled) ---------------------------------
def s_volmanaged_mkt():
    d = ffd()["Mkt-RF"]
    vol = d.groupby(pd.Grouper(freq="ME")).std() * np.sqrt(252)
    w = (0.16 / vol).clip(upper=2.0).shift(1)
    mx = s_mkt()
    out = (w.reindex(mx.index) * mx).dropna()
    # turnover cost on index futures/ETF: 5 bp per unit weight change
    return out - w.reindex(out.index).diff().abs().fillna(0) * ETF_COST


def s_volmanaged_mom():
    d = D.french("F-F_Momentum_Factor_daily")["Mom"]
    vol = d.rolling(126).std().resample("ME").last() * np.sqrt(252)
    w = (0.12 / vol).clip(upper=2.0).shift(1)
    m = umd()
    return (w.reindex(m.index) * m).dropna()


# --- 5. Market timing on the long US record (French market, 1926+) -------
def _timing(signal_on: pd.Series) -> pd.Series:
    """signal_on known at end of t -> invested in market in t+1, else T-bill."""
    f = ff3()
    on = signal_on.shift(1).reindex(f.index)
    valid = on.notna()
    on = on[valid].astype(float)
    ex = on * f["Mkt-RF"][valid]
    switches = on.diff().abs().fillna(0)
    return (ex - switches * ETF_COST).dropna()


def s_sma10():
    px = (1 + mkt_total_m()).cumprod()
    sma = px.rolling(10).mean()
    return _timing((px > sma).where(sma.notna()))


def s_tsmom_mkt():
    ex = ff3()["Mkt-RF"]
    past = ex.rolling(12).sum()
    return _timing((past > 0).where(past.notna()))


def s_sellinmay():
    f = ff3()
    on = pd.Series(f.index.month.isin([11, 12, 1, 2, 3, 4]), index=f.index)
    # month-of-year is known in advance; shift(-1) so _timing's shift lines up
    return _timing(on.shift(-1).dropna())


def s_january_smb():
    smb = ff3()["SMB"]
    return smb.where(smb.index.month == 1, 0.0)


# --- 6. Calendar effects on daily data (French daily market, 1926+) ------
def _daily_timing(on: pd.Series) -> pd.Series:
    d = ffd()
    on = on.reindex(d.index).fillna(False).astype(float)
    ex = on * d["Mkt-RF"]
    trades = on.diff().abs().fillna(0)
    ex = ex - trades * DAY_COST
    tot = ex + d["RF"]
    return to_monthly_excess(tot, d["RF"])


def s_turnofmonth():
    idx = ffd().index
    s = pd.Series(idx, index=idx)
    ym = idx.to_period("M")
    pos = s.groupby(ym).cumcount()                    # 0 = first trading day
    rpos = s.groupby(ym).cumcount(ascending=False)    # 0 = last trading day
    on = (pos <= 2) | (rpos == 0)
    return _daily_timing(on)


def s_nomonday():
    idx = ffd().index
    return _daily_timing(pd.Series(idx.dayofweek != 0, index=idx))


def s_preholiday():
    idx = ffd().index
    nxt = pd.Series(idx[1:].append(pd.DatetimeIndex([idx[-1] + pd.Timedelta(days=1)])), index=idx)
    gap_bdays = np.busday_count(idx.values.astype("datetime64[D]"),
                                nxt.values.astype("datetime64[D]"))
    # pre-1952 Saturday sessions make busday gaps misleading; start 1953
    on = pd.Series(gap_bdays > 1, index=idx) & (idx.year >= 1953)
    return _daily_timing(on)


def s_overnight():
    d = D.etf_ohlc("SPY")
    on = d["Open"] / d["Close"].shift(1) - 1
    net = on - 2 * DAY_COST  # buy at close, sell at next open
    return to_monthly_excess(net.dropna(), D.rf_daily()) + 0  # cash intraday


# --- 7. Tactical asset allocation on ETFs --------------------------------
def _etf_monthly_rets(tickers):
    px = month_end_prices(tickers)
    return px, px.pct_change()


def s_gtaa5():
    t = ["SPY", "EFA", "IEF", "VNQ", "DBC"]
    px, r = _etf_monthly_rets(t)
    sma = px.rolling(10).mean()
    w = (px > sma).astype(float) / len(t)
    w = w[sma.notna().all(axis=1)]
    return weights_backtest(w, r, rf_m_for(r.index))


def s_gem():
    t = ["SPY", "EFA", "AGG"]
    px, r = _etf_monthly_rets(t)
    rf = rf_m_for(r.index)
    r12 = px.pct_change(12)
    rf12 = (1 + rf).rolling(12).apply(np.prod, raw=True) - 1
    w = pd.DataFrame(0.0, index=px.index, columns=t)
    ok = r12[["SPY", "EFA"]].notna().all(axis=1) & px["AGG"].notna()
    for dt in px.index[ok]:
        if r12.at[dt, "SPY"] > rf12.at[dt]:
            w.at[dt, "SPY" if r12.at[dt, "SPY"] >= r12.at[dt, "EFA"] else "EFA"] = 1.0
        else:
            w.at[dt, "AGG"] = 1.0
    return weights_backtest(w[ok], r, rf)


def s_sector_rot():
    t = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
    px, r = _etf_monthly_rets(t)
    r12 = px.pct_change(12)
    w = pd.DataFrame(0.0, index=px.index, columns=t)
    for dt in r12.dropna().index:
        w.loc[dt, r12.loc[dt].nlargest(3).index] = 1 / 3
    return weights_backtest(w.loc[r12.dropna().index], r, rf_m_for(r.index))


def s_riskparity():
    t = ["SPY", "EFA", "TLT", "IEF", "GLD", "DBC", "VNQ"]
    daily = D.etf(t).pct_change()
    vol = daily.rolling(63).std().resample("ME").last()
    inv = 1 / vol
    w = inv.div(inv.sum(axis=1), axis=0)
    w = w[vol.notna().all(axis=1)]
    px, r = _etf_monthly_rets(t)
    return weights_backtest(w, r, rf_m_for(r.index))


def s_6040():
    t = ["SPY", "IEF"]
    px, r = _etf_monthly_rets(t)
    w = pd.DataFrame({"SPY": 0.6, "IEF": 0.4}, index=px.index)[px.notna().all(axis=1)]
    return weights_backtest(w, r, rf_m_for(r.index))


def s_vix_term():
    px = D.etf(["SPY", "^VIX", "^VIX3M"]).dropna()
    on = (px["^VIX"] < px["^VIX3M"]).astype(float).shift(1).dropna()
    r = px["SPY"].pct_change().reindex(on.index)
    rf = D.rf_daily().reindex(on.index).ffill()
    tot = on * r + (1 - on) * rf - on.diff().abs().fillna(0) * DAY_COST
    return to_monthly_excess(tot.dropna(), D.rf_daily())


# ================================================================ REGISTRY ==
def _q(s: str) -> str:
    return "https://papers.ssrn.com/sol3/results.cfm?txtKey_Words=" + s.replace(" ", "+")


R = []


def reg(id, name, fn, cat, paper, authors, year, rule, costs, data, benchmark=False):
    R.append(dict(id=id, name=name, fn=fn, category=cat, paper=paper, authors=authors,
                  year=year, rule=rule, costs=costs, data=data, benchmark=benchmark,
                  link=_q(paper)))


FR = "French Data Library (CRSP)"
GROSS = "Gross of costs (published portfolio)"

reg("mkt", "US equity market", s_mkt, "Benchmark", "Common Risk Factors in the Returns on Stocks and Bonds",
    "Fama & French", 1993, "Hold the value-weighted CRSP market.", GROSS, FR, benchmark=True)
reg("6040", "60/40 SPY / IEF", s_6040, "Benchmark", "60/40 portfolio", "—", 2002,
    "60% SPY, 40% IEF, rebalanced monthly.", "5 bp per unit turnover", "Yahoo ETFs", benchmark=True)

reg("smb", "Size (small minus big)", s_smb, "Equity anomaly", "The Relationship between Return and Market Value of Common Stocks",
    "Banz", 1981, "Long small caps, short large caps (FF SMB, 2x3 sorts).", GROSS, FR)
reg("hml", "Value (book-to-market)", s_hml, "Equity anomaly", "The Cross-Section of Expected Stock Returns",
    "Fama & French", 1992, "Long high B/M, short low B/M (FF HML).", GROSS, FR)
reg("ep", "Value (earnings yield)", s_ep, "Equity anomaly", "Investment Performance of Common Stocks in Relation to Their Price-Earnings Ratios",
    "Basu", 1977, "Long top-quintile E/P, short bottom quintile, value-weighted.", GROSS, FR)
reg("cfp", "Value (cash-flow yield)", s_cfp, "Equity anomaly", "Contrarian Investment, Extrapolation, and Risk",
    "Lakonishok, Shleifer & Vishny", 1994, "Long top-quintile CF/P, short bottom quintile.", GROSS, FR)
reg("dp", "Dividend yield", s_dp, "Equity anomaly", "The Effect of Personal Taxes and Dividends on Capital Asset Prices",
    "Litzenberger & Ramaswamy", 1979, "Long top-quintile D/P, short bottom quintile.", GROSS, FR)
reg("umd", "Momentum (12-1)", s_umd, "Equity anomaly", "Returns to Buying Winners and Selling Losers",
    "Jegadeesh & Titman", 1993, "Long past 12-1 month winners, short losers (FF UMD).", GROSS + "; high turnover", FR)
reg("str", "Short-term reversal", s_str, "Equity anomaly", "Evidence of Predictable Behavior of Security Returns",
    "Jegadeesh", 1990, "Long last month's losers, short last month's winners.", GROSS + "; very high turnover", FR)
reg("ltr", "Long-term reversal", s_ltr, "Equity anomaly", "Does the Stock Market Overreact?",
    "De Bondt & Thaler", 1985, "Long 5-year (t-60..t-13) losers, short winners.", GROSS, FR)
reg("rmw", "Profitability", s_rmw, "Equity anomaly", "The Other Side of Value: The Gross Profitability Premium",
    "Novy-Marx", 2013, "Long robust operating profitability, short weak (FF RMW).", GROSS, FR)
reg("cma", "Investment (asset growth)", s_cma, "Equity anomaly", "Capital Investments and Stock Returns",
    "Titman, Wei & Xie", 2004, "Long conservative investors, short aggressive (FF CMA).", GROSS, FR)
reg("accr", "Accruals", s_accruals, "Equity anomaly", "Do Stock Prices Fully Reflect Information in Accruals and Cash Flows About Future Earnings?",
    "Sloan", 1996, "Long lowest-accrual quintile, short highest.", GROSS, FR)
reg("ni", "Net share issuance", s_netissue, "Equity anomaly", "Share Issuance and Cross-Sectional Returns",
    "Pontiff & Woodgate", 2008, "Long net repurchasers, short top-quintile issuers.", GROSS, FR)
reg("lowbeta", "Low beta (raw)", s_lowbeta, "Equity anomaly", "The Capital Asset Pricing Model: Some Empirical Tests",
    "Black, Jensen & Scholes", 1972, "Long lowest-beta quintile, short highest (dollar-neutral, not beta-neutral).", GROSS, FR)
reg("lowvar", "Low volatility", s_lowvar, "Equity anomaly", "The Cross-Section of Volatility and Expected Returns",
    "Ang, Hodrick, Xing & Zhang", 2006, "Long lowest total-variance quintile, short highest.", GROSS, FR)
reg("lowivol", "Low idiosyncratic volatility", s_lowivol, "Equity anomaly", "The Cross-Section of Volatility and Expected Returns",
    "Ang, Hodrick, Xing & Zhang", 2006, "Long lowest residual-variance quintile, short highest.", GROSS, FR)
reg("valmom", "Value + momentum combo", s_valmom, "Equity anomaly", "Value and Momentum Everywhere",
    "Asness, Moskowitz & Pedersen", 2013, "50% HML + 50% UMD.", GROSS, FR)
reg("bab", "Betting against beta", s_bab, "Equity anomaly", "Betting Against Beta",
    "Frazzini & Pedersen", 2014, "Long levered low-beta, short de-levered high-beta; beta-neutral (AQR, US).", GROSS, "AQR Data Library")
reg("qmj", "Quality minus junk", s_qmj, "Equity anomaly", "Quality Minus Junk",
    "Asness, Frazzini & Pedersen", 2019, "Long profitable, growing, safe, high-payout firms; short junk (AQR, US).", GROSS, "AQR Data Library")

reg("indmom", "Industry momentum", s_indmom, "Rotation", "Do Industries Explain Momentum?",
    "Moskowitz & Grinblatt", 1999, "Rank 49 industries on 6-month return; long top 7, short bottom 7; monthly.", "Gross; built from French industry returns", FR)
reg("indseas", "Return seasonality", s_indseason, "Rotation", "Seasonality in the Cross-Section of Stock Returns",
    "Heston & Sadka", 2008, "Industry-level version: rank 49 industries on their average return in the same calendar month over the prior 20 years; long top 7, short bottom 7.", "Gross; built from French industry returns", FR)
reg("facmom", "Factor momentum", s_factormom, "Rotation", "Factor Momentum and the Momentum Factor",
    "Ehsani & Linnainmaa", 2022, "Each month, long each of 7 FF factors with positive trailing 12-month return, short those with negative; equal-weight.", "Gross", FR)
reg("vme", "Value & momentum everywhere (global)", s_vme, "Multi-asset", "Value and Momentum Everywhere",
    "Asness, Moskowitz & Pedersen", 2013, "50/50 value and momentum across 8 asset classes (AQR 'Everywhere').", GROSS, "AQR Data Library")
reg("tsmom", "Time-series momentum (diversified)", s_tsmom_aqr, "Multi-asset", "Time Series Momentum",
    "Moskowitz, Ooi & Pedersen", 2012, "Long futures with positive 12-month return, short negative; vol-scaled, 58 markets (AQR).", "Gross; futures", "AQR Data Library")

reg("vmmkt", "Volatility-managed market", s_volmanaged_mkt, "Risk-managed", "Volatility-Managed Portfolios",
    "Moreira & Muir", 2017, "Scale market exposure by 16% / last month's realized vol, capped at 2x.", "5 bp per unit weight change", FR + " (daily)")
reg("vmmom", "Volatility-managed momentum", s_volmanaged_mom, "Risk-managed", "Momentum Has Its Moments",
    "Barroso & Santa-Clara", 2015, "Scale UMD to 12% target vol using prior 6-month daily vol, capped at 2x.", GROSS, FR + " (daily)")

reg("sma10", "10-month moving-average timing", s_sma10, "Market timing", "A Quantitative Approach to Tactical Asset Allocation",
    "Faber", 2007, "Hold US market when total-return index > 10-month SMA, else T-bills.", "5 bp per switch", FR)
reg("tsmkt", "Absolute momentum on the market", s_tsmom_mkt, "Market timing", "Absolute Momentum: A Simple Rule-Based Strategy",
    "Antonacci", 2013, "Hold US market when trailing 12-month excess return > 0, else T-bills.", "5 bp per switch", FR)
reg("halloween", "Sell in May (Halloween)", s_sellinmay, "Calendar", "The Halloween Indicator, 'Sell in May and Go Away'",
    "Bouman & Jacobsen", 2002, "Hold US market November-April, T-bills May-October.", "5 bp per switch", FR)
reg("jan", "January small-cap effect", s_january_smb, "Calendar", "Size-Related Anomalies and Stock Return Seasonality",
    "Keim", 1983, "Hold SMB in January only; flat the rest of the year.", GROSS, FR)
reg("tom", "Turn of the month", s_turnofmonth, "Calendar", "Equity Returns at the Turn of the Month",
    "McConnell & Xu", 2008, "Hold US market on the last trading day of the month and the first 3 of the next; T-bills otherwise.", "1 bp per side", FR + " (daily)")
reg("monday", "Avoid Mondays (weekend effect)", s_nomonday, "Calendar", "Stock Returns and the Weekend Effect",
    "French", 1980, "Hold US market every day except Monday.", "1 bp per side", FR + " (daily)")
reg("preh", "Pre-holiday effect", s_preholiday, "Calendar", "High Stock Returns before Holidays",
    "Ariel", 1990, "Hold US market only on the trading day before a market holiday (1953+).", "1 bp per side", FR + " (daily)")
reg("overnight", "Overnight drift (SPY)", s_overnight, "Calendar", "Return Differences between Trading and Non-Trading Hours",
    "Cooper, Cliff & Gulen", 2008, "Buy SPY at the close, sell at the next open; cash intraday.", "1 bp per side, every day", "Yahoo SPY OHLC")

reg("gtaa5", "GTAA 5-asset trend (Faber)", s_gtaa5, "Tactical allocation", "A Quantitative Approach to Tactical Asset Allocation",
    "Faber", 2007, "20% each in SPY, EFA, IEF, VNQ, DBC when above 10-month SMA, else that sleeve in T-bills.", "5 bp per unit turnover", "Yahoo ETFs")
reg("gem", "Global equities momentum (dual momentum)", s_gem, "Tactical allocation", "Risk Premia Harvesting Through Dual Momentum",
    "Antonacci", 2012, "If SPY 12m > T-bills, hold the stronger of SPY/EFA; else AGG.", "5 bp per unit turnover", "Yahoo ETFs")
reg("sector", "Sector momentum rotation", s_sector_rot, "Tactical allocation", "Relative Strength Strategies for Investing",
    "Faber", 2010, "Hold the 3 of 9 SPDR sectors with the best 12-month return, equal weight.", "5 bp per unit turnover", "Yahoo ETFs")
reg("rp", "Risk parity (inverse vol)", s_riskparity, "Tactical allocation", "Leverage Aversion and Risk Parity",
    "Asness, Frazzini & Pedersen", 2012, "Inverse 63-day-vol weights across SPY, EFA, TLT, IEF, GLD, DBC, VNQ; unlevered; monthly.", "5 bp per unit turnover", "Yahoo ETFs")
