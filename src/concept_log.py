"""The reinvention log: every concept tested, its best development result, and
why it lived or died.

Numbers are the best configuration found for that concept on the 2025
development window, measured with the prop-firm limits switched off so a single
early breach could not disguise a real edge. "expR" is expectancy per trade in
units of the risk taken on that trade.
"""

CONCEPTS = [
    # (name, family, trades, expR, pf, verdict, note)
    ("Asian range breakout", "breakout", 115, 0.018, 1.04, "rejected",
     "Gold fakes out of the overnight range constantly. Barely above break-even "
     "before costs."),
    ("H1 trend pullback (tight stops)", "trend", 159, -0.038, 0.93, "reworked",
     "Losing at a 1.2x ATR stop. Survived as a concept only because widening the "
     "stop later flipped it positive."),
    ("H1 Donchian channel", "trend", 80, 0.198, 1.43, "promoted",
     "The first concept with a clean edge. Became account A1 after tuning."),
    ("Range fade (mean reversion)", "reversion", 94, -0.126, 0.79, "rejected",
     "Fading gold in a trending year loses steadily."),
    ("Daily Donchian swing", "trend", 0, 0.0, 0.0, "infeasible",
     "A 2x daily-ATR stop is $85-200 wide. Correct sizing needs less than the "
     "0.01 lot minimum, so a $6K account cannot trade it at all."),
    ("Asian compression breakout", "breakout", 61, -0.026, 0.91, "rejected",
     "Negative at every parameter setting tried, even with the compression "
     "filter and a trend alignment requirement."),
    ("H1 trend pullback (wide stops)", "trend", 106, 0.318, 1.68, "promoted",
     "The same concept as above with a 2.0x ATR stop. Became account A2."),
    ("New York momentum continuation", "breakout", 137, 0.083, 1.24, "rejected",
     "Positive but thin. Attractively low drawdown, not enough edge to clear "
     "an 8% target."),
    ("Range fade v2", "reversion", 37, -0.137, 0.61, "rejected",
     "Retested with the wide-stop lesson applied. Still negative everywhere."),
    ("Displacement from daily open", "breakout", 98, 0.098, 1.37, "rejected",
     "Respectable on paper, never once got funded across 28 simulated "
     "challenge starts."),
    ("Keltner band expansion", "trend", 90, 0.258, 1.48, "rejected",
     "Genuine edge, but passed Phase 1 in only 64% of starts and got funded in "
     "at most 18%. Edge and rule survival are different questions."),
    ("Failed-breakout reversal", "reversion", 197, -0.112, 0.82, "rejected",
     "Built specifically as a decorrelation play. Negative at every setting: "
     "counter-trend simply did not pay on gold in this period."),
    ("H4 Donchian channel", "trend", 77, 0.452, 1.98, "promoted",
     "Strongest concept tested. Became account A3."),
    ("Previous-day high/low break", "breakout", 139, 0.113, 1.25, "rejected",
     "A level everyone watches, and priced accordingly."),
    ("Multi-timeframe aligned momentum", "trend", 49, 0.455, 2.00, "promoted",
     "Only trades when H1, H4 and D1 agree. Flat most of the time. Became "
     "account A4."),
]

SUMMARY = {
    "tested": len(CONCEPTS),
    "promoted": sum(1 for c in CONCEPTS if c[5] == "promoted"),
    "rejected": sum(1 for c in CONCEPTS if c[5] in ("rejected", "infeasible")),
}

if __name__ == "__main__":
    print(f"{'concept':36s} {'fam':>10s} {'trades':>7s} {'expR':>7s} {'PF':>6s}  verdict")
    for name, fam, n, e, pf, verdict, _ in CONCEPTS:
        print(f"{name:36s} {fam:>10s} {n:7d} {e:7.3f} {pf:6.2f}  {verdict}")
    print(f"\n{SUMMARY['tested']} concepts tested, {SUMMARY['promoted']} promoted, "
          f"{SUMMARY['rejected']} discarded")
