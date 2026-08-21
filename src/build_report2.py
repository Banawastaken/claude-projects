"""Report: which strategy is safest, and which CFDs it actually works on."""

from __future__ import annotations

import base64
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")

# Monte Carlo of all four strategies on gold M1, current engine.
# 19 development starts x 205 days, 17 test starts x 175 days.
STRATS = [
    # name, P1 dev, fund dev, P1 test, fund test, alive test, breach, ddMed test,
    # worst-day median test, verdict
    ("A1", "Donchian H1 Breakout", 100, 100, 59, 18, 18, 0, 13.23, 2.92,
     "Best in sample, worst out of it. Classic overfit signature."),
    ("A2", "H1 Trend Pullback", 89, 68, 100, 35, 35, 0, 9.45, 2.28,
     "Steady but slow; rarely clears the target inside the horizon."),
    ("A3", "Donchian H4 Swing", 89, 63, 100, 88, 88, 0, 9.64, 1.52,
     "The pick. Never breached, best out-of-sample funding rate, mildest days."),
    ("A4", "Multi-Timeframe Aligned", 89, 42, 65, 47, 47, 0, 4.15, 1.93,
     "Shallowest drawdown of the four, but too selective to fund reliably."),
]


def img(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build():
    edge = pd.read_csv(os.path.join(REPORTS, "instrument_edge.csv"))
    res = pd.read_csv(os.path.join(REPORTS, "instrument_results.csv"))
    piv = res.pivot_table(index="instrument", columns="window",
                          values=["funded", "breach", "p1_pass"])
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]

    edge["both_pos"] = (edge["exp_DEV"] > 0) & (edge["exp_TEST"] > 0)
    edge = edge.sort_values("exp_avg", ascending=False)

    # ---- strategy comparison table --------------------------------------
    srows = []
    for tag, name, p1d, fd, p1t, ft, at, br, dd, wd, note in STRATS:
        pick = tag == "A3"
        srows.append(
            f"<tr class='{'pick' if pick else ''}'>"
            f"<td><span class='tag'>{tag}</span> {esc(name)}</td>"
            f"<td class='n'>{p1d}</td><td class='n'>{fd}</td>"
            f"<td class='n'>{p1t}</td><td class='n {'pos' if ft >= 80 else ''}'>{ft}</td>"
            f"<td class='n'>{at}</td><td class='n {'pos' if br == 0 else 'neg'}'>{br}</td>"
            f"<td class='n'>{dd:.1f}</td>"
            f"<td class='n {'pos' if wd < 2 else ''}'>{wd:.2f}</td>"
            f"<td class='note'>{esc(note)}</td></tr>")
    strat_table = (
        "<table><thead><tr><th>Strategy</th>"
        "<th class='n'>P1 dev</th><th class='n'>funded dev</th>"
        "<th class='n'>P1 test</th><th class='n'>funded test</th>"
        "<th class='n'>alive test</th><th class='n'>breach</th>"
        "<th class='n'>DD med</th><th class='n'>worst day</th>"
        f"<th>Read</th></tr></thead><tbody>{''.join(srows)}</tbody></table>")

    # ---- instrument table -------------------------------------------------
    irows = []
    for _, r in edge.iterrows():
        name = r["instrument"]
        fd = piv.loc[name, "funded_DEV"] if name in piv.index else float("nan")
        ft = piv.loc[name, "funded_TEST"] if name in piv.index else float("nan")
        br = max(piv.loc[name, "breach_DEV"], piv.loc[name, "breach_TEST"]) \
            if name in piv.index else 0.0
        ok = bool(r["both_pos"])
        irows.append(
            f"<tr class='{'good' if ok else ''}'>"
            f"<td>{'&#9733; ' if ok else ''}{esc(name)}</td>"
            f"<td class='muted'>{esc(r['class'])}</td>"
            f"<td class='n {'pos' if r['exp_DEV'] > 0 else 'neg'}'>{r['exp_DEV']:+.3f}</td>"
            f"<td class='n {'pos' if r['exp_TEST'] > 0 else 'neg'}'>{r['exp_TEST']:+.3f}</td>"
            f"<td class='n'>{r['totR_sum']:+.1f}</td>"
            f"<td class='n'>{fd:.0f}</td><td class='n'>{ft:.0f}</td>"
            f"<td class='n {'neg' if br > 10 else ''}'>{br:.0f}</td>"
            f"<td class='n'>{r['spread_bp']:.2f}</td></tr>")
    inst_table = (
        "<table><thead><tr><th>Instrument</th><th>Class</th>"
        "<th class='n'>Exp R dev</th><th class='n'>Exp R test</th>"
        "<th class='n'>Total R</th><th class='n'>funded dev %</th>"
        "<th class='n'>funded test %</th><th class='n'>breach %</th>"
        f"<th class='n'>spread bp</th></tr></thead><tbody>{''.join(irows)}</tbody></table>")

    winners = edge[edge["both_pos"]]["instrument"].tolist()
    n_pos = len(winners)
    chart = img(os.path.join(REPORTS, "instrument_edge.png"))

    return TEMPLATE.format(
        strat_table=strat_table,
        inst_table=inst_table,
        chart=(f'<figure><img src="{chart}" alt="expectancy by instrument">'
               f"<figcaption>Every instrument, both windows. Only the starred five are "
               f"positive in each &mdash; and outside gold, only barely.</figcaption>"
               f"</figure>" if chart else ""),
        n_tested=len(edge),
        n_pos=n_pos,
        others=", ".join(w for w in winners if w != "XAUUSD"),
    )


TEMPLATE = """<title>Gold or Nothing</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root{{
  --paper:#F7F5F0; --raised:#FFFDFA; --sunk:#EFEBE2;
  --ink:#16140F; --muted:#6E6A60; --rule:#E2DDD2;
  --accent:#A8791C; --pass:#1E6F4C; --fail:#A6301F; --warn:#8A6212;
  --shadow:0 1px 2px rgba(22,20,15,.06),0 8px 24px -12px rgba(22,20,15,.18);
  --display:'Archivo',system-ui,sans-serif;
  --body:'Source Serif 4',Georgia,serif;
  --mono:'IBM Plex Mono',ui-monospace,monospace;
}}
@media (prefers-color-scheme:dark){{
  :root:not([data-theme="light"]){{
    --paper:#14130E; --raised:#1C1A14; --sunk:#100F0B;
    --ink:#E9E4D9; --muted:#9A948A; --rule:#2C2921;
    --accent:#D9AC46; --pass:#4FA37A; --fail:#D9705F; --warn:#C9963A;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
  }}
}}
:root[data-theme="dark"]{{
  --paper:#14130E; --raised:#1C1A14; --sunk:#100F0B;
  --ink:#E9E4D9; --muted:#9A948A; --rule:#2C2921;
  --accent:#D9AC46; --pass:#4FA37A; --fail:#D9705F; --warn:#C9963A;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px -14px rgba(0,0,0,.7);
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:var(--body);
  font-size:17px;line-height:1.65;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 24px 96px}}
.prose{{max-width:68ch}}
h1,h2,h3{{font-family:var(--display);text-wrap:balance;line-height:1.15;margin:0}}
h1{{font-size:clamp(2.1rem,5vw,3.4rem);font-weight:700;letter-spacing:-.022em}}
h2{{font-size:1.6rem;font-weight:600;letter-spacing:-.012em;margin:0 0 .6rem}}
h3{{font-size:1.06rem;font-weight:600}}
p{{margin:0 0 1.05rem}}
.muted{{color:var(--muted)}}
.eyebrow{{font-family:var(--mono);font-size:.72rem;font-weight:500;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 1rem}}
header.top{{padding:72px 0 40px;border-bottom:1px solid var(--rule)}}
.dek{{font-size:1.22rem;color:var(--muted);max-width:60ch;margin-top:1.1rem}}
.meta{{display:flex;flex-wrap:wrap;gap:28px;margin-top:32px;
  font-family:var(--mono);font-size:.78rem;color:var(--muted)}}
.meta b{{display:block;color:var(--ink);font-size:1.32rem;font-weight:600;
  letter-spacing:-.01em;font-variant-numeric:tabular-nums}}
section{{padding:56px 0 0}}
.section-head{{margin-bottom:26px}}
.section-head .eyebrow{{margin-bottom:.5rem}}
.answer{{border-left:3px solid var(--accent);background:var(--raised);
  padding:20px 24px;border-radius:0 3px 3px 0;margin:0 0 24px;box-shadow:var(--shadow)}}
.answer h3{{margin-bottom:.45rem;font-size:1.15rem}}
.answer p:last-child{{margin-bottom:0}}
.answer.hard{{border-left-color:var(--fail)}}
.tablewrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;
  background:var(--raised);box-shadow:var(--shadow)}}
table{{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.78rem}}
th,td{{padding:8px 11px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  font-weight:600;background:var(--sunk);position:sticky;top:0}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:last-child td{{border-bottom:none}}
tr.pick td{{background:color-mix(in srgb,var(--accent) 9%,transparent);font-weight:600}}
tr.good td{{background:color-mix(in srgb,var(--pass) 8%,transparent)}}
.pos{{color:var(--pass)}} .neg{{color:var(--fail)}}
.tag{{font-family:var(--mono);font-size:.68rem;font-weight:600;color:var(--accent);
  border:1px solid var(--rule);border-radius:2px;padding:1px 5px;margin-right:5px}}
td.note{{white-space:normal;min-width:30ch;font-family:var(--body);font-size:.85rem;
  color:var(--muted)}}
figure{{margin:0 0 26px}}
figure img{{width:100%;height:auto;display:block;border:1px solid var(--rule);
  border-radius:3px;background:#0d1117}}
figcaption{{font-size:.86rem;color:var(--muted);margin-top:9px;max-width:70ch}}
ul.plain{{padding-left:1.1rem}}
ul.plain li{{margin-bottom:.6rem}}
hr{{border:none;border-top:1px solid var(--rule);margin:0}}
footer{{padding:44px 0 0;color:var(--muted);font-size:.86rem}}
@media (max-width:640px){{body{{font-size:16px}}.wrap{{padding:0 16px 64px}}
  header.top{{padding:44px 0 30px}}.meta{{gap:18px}}}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">FundedNext Stellar 2-Step $6K &middot; one strategy, {n_tested} CFDs</p>
  <h1>Gold or nothing</h1>
  <p class="dek">The safest of the four strategies was run across every CFD FundedNext
  lists. It is not a trend system that happens to trade gold. It is a gold system.</p>
  <div class="meta">
    <div>Instruments tested<b>{n_tested}</b></div>
    <div>Positive in both windows<b>{n_pos}</b></div>
    <div>Worth trading<b>1</b></div>
    <div>Strategy chosen<b>A3</b></div>
  </div>
</header>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Question one</p>
    <h2>The most consistent strategy is A3</h2>
  </div>
  <div class="prose">
    <div class="answer">
      <h3>A3 Donchian H4 Swing &mdash; and it is not the one with the smallest drawdown.</h3>
      <p>A3 never breached an account in any simulated challenge, in either window.
      It passed Phase 1 in 100% of out-of-sample starts and got funded in 88% of them,
      against 18&ndash;47% for the others, and its median worst day was 1.52% against
      the 5% daily limit &mdash; the widest safety margin of the four.</p>
    </div>
    <p>A4 has the shallowest drawdown (4.15% median against A3's 9.64%), and on that
    number alone it looks like the safer account. It is not: it only reaches the funded
    stage in 42&ndash;47% of starts, because it is so selective it rarely clears the
    profit target inside the horizon. An account that never gets funded is not safe,
    it is just idle.</p>
    <p>A1 is the trap. It is the best strategy in the development window &mdash; 100%
    funded, every start &mdash; and one of the worst outside it, at 18%. That gap
    between in-sample and out-of-sample is what overfitting looks like from the
    outside.</p>
  </div>
  <div class="tablewrap">{strat_table}</div>
  <div class="prose" style="margin-top:20px">
    <p class="muted">Percentages across 19 development starts (205 days each) and 17
    test starts (175 days each), on gold minute data. "Alive" means the funded account
    was still open at the end of the run.</p>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Question two</p>
    <h2>Which instruments it works on</h2>
  </div>
  <div class="prose">
    <div class="answer hard">
      <h3>Trade it on gold. There is no credible second instrument.</h3>
      <p>Of {n_tested} CFDs, only {n_pos} were profitable in both windows, and gold's
      edge is roughly four times the next one's. The others &mdash; {others} &mdash;
      are positive but far too weak to clear an 8% target before the drawdown limit
      finds them: none of them got funded in even one simulated start.</p>
    </div>
  </div>
  {chart}
  <div class="tablewrap">{inst_table}</div>
  <div class="prose" style="margin-top:22px">
    <p>Two entries look tempting and are worse than useless. NDX100 returns +0.404 R
    per trade in 2025 and &minus;0.493 in the test window; silver goes +0.304 then
    &minus;0.219. Averaging those to something mildly positive would be exactly the
    wrong conclusion &mdash; they are not edges, they are one good regime each.</p>
    <p>Three instruments were also actively dangerous. Hong Kong 50 breached the
    account in 63% of development starts, Ethereum in 42%, Japan 225 in 37%, against
    0% for gold. Index and crypto CFDs gap across their closed hours, and a gap is the
    one thing a stop cannot protect you from.</p>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Why</p>
    <h2>Why only gold</h2>
  </div>
  <div class="prose">
    <p>The strategy buys a break of the 20-bar four-hour range with a stop one ATR
    wide. That needs a market that trends cleanly at the four-hour scale and keeps
    going once it breaks out. Gold in 2025&ndash;26 did exactly that, twice, in both
    directions.</p>
    <p>Major FX did not. EURUSD, USDCAD and the rest spent the period range-bound at
    that horizon, so a channel break is mostly noise: the edge is positive but tiny,
    around +0.09 R against gold's +0.58. Equity indices trend, but they trend
    <i>overnight</i>, in gaps that a session-hours breakout system never gets to
    participate in while still eating the gap risk on the wrong side.</p>
    <p>The uncomfortable implication is that this was never a portfolio of strategies
    and cannot become one by adding instruments. It is one edge, on one market, and
    the honest way to run it is one account.</p>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">What I would actually do</p>
    <h2>Recommendation</h2>
  </div>
  <div class="prose">
    <ul class="plain">
      <li><b>One gold account running A3.</b> Not four accounts, and not four
      instruments. Everything this test found says the extra accounts buy correlated
      risk and extra fees, not diversification.</li>
      <li><b>0.75% risk per trade, not 1%.</b> At 1% the H4 channel gets funded
      slightly more often and then dies: 16&ndash;53% of funded accounts survived at
      1%, against 82&ndash;100% at 0.75%.</li>
      <li><b>If you want a second account, buy a second gold account, not a second
      instrument.</b> Two uncorrelated bad edges are worse than one good one.</li>
      <li><b>Do not read the second tier as a shortlist.</b> JP225, USDCAD, EURUSD and
      GER30 are positive, but at roughly a sixth of gold's edge they cannot outrun the
      cost of a failed challenge.</li>
    </ul>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Caveats</p>
    <h2>What would change this</h2>
  </div>
  <div class="prose">
    <div class="answer">
      <h3>The cross-instrument screen runs on hourly bars, and hourly bars flatter it</h3>
      <p>Minute data for thirty symbols is thousands of downloads, so the sweep uses
      hourly candles. On gold, where both exist, the hourly version reports +0.620 R
      per trade in the test window against +0.264 on minute data, because the trailing
      stop ratchets once an hour instead of once a minute and winners run further.
      Every instrument here is therefore flattered, which makes the negative results
      more damning rather than less.</p>
    </div>
    <div class="answer">
      <h3>One strategy, one period</h3>
      <p>This says A3's particular edge lives in gold over these twenty months. It does
      not say gold is the best instrument in general, nor that FX and indices are
      untradeable &mdash; only that this breakout logic does not pay on them here.</p>
    </div>
    <div class="answer">
      <h3>Index CFDs are priced in their home currency</h3>
      <p>GER30 settles in euros, JP225 in yen. A constant conversion rate is used, which
      cancels out of every R multiple and reaches the result only through commission and
      the minimum-lot check.</p>
    </div>
  </div>
</section>

<footer>
  <hr>
  <p style="margin-top:22px">Dukascopy bid/ask data, Jan 2025 &ndash; Aug 2026.
  FundedNext contract sizes and commissions ($5/lot forex and oil, 0.0016% notional on
  metals, 0.04% on crypto). Simulated results from historical data; not a prediction and
  not investment advice.</p>
</footer>
</div>
"""


if __name__ == "__main__":
    html = build()
    out = os.path.join(REPORTS, "report_instruments.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
