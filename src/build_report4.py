"""Report on the second search: a null result, and what it is worth."""

from __future__ import annotations

import base64
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPORTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports")

SIG = [
    ("Turn of month", "calendar / flows", 641, 0.055, 0.043, 1.73, 52),
    ("Time-series momentum", "cross-asset momentum", 2317, 0.062, 0.010, 1.93, 50),
    ("Short-term reversal", "mean reversion", 2247, 0.038, 0.004, 1.38, 44),
    ("A3 Donchian (control)", "breakout", 4267, 0.046, -0.034, 1.87, 43),
    ("Volatility contraction", "volatility", 1505, -0.031, -0.078, -0.90, 33),
]


def img(p):
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build():
    rows = "".join(
        f"<tr><td>{esc(n)}</td><td class='muted'>{esc(fam)}</td>"
        f"<td class='n'>{cnt:,}</td>"
        f"<td class='n {'pos' if g > 0 else 'neg'}'>{g:+.3f}</td>"
        f"<td class='n {'pos' if net > 0 else 'neg'}'>{net:+.3f}</td>"
        f"<td class='n'>{t:+.2f}</td>"
        f"<td class='n {'pos' if br >= 55 else 'neg'}'>{br}%</td>"
        f"<td class='n neg'>no</td></tr>"
        for n, fam, cnt, g, net, t, br in SIG)
    table = ("<table><thead><tr><th>Concept</th><th>Family</th><th class='n'>Trades</th>"
             "<th class='n'>Gross R</th><th class='n'>Net R</th><th class='n'>t-stat</th>"
             "<th class='n'>Breadth</th><th class='n'>Significant?</th></tr></thead>"
             f"<tbody>{rows}</tbody></table>")
    chart = img(os.path.join(REPORTS, "edge_gap.png"))
    return TEMPLATE.format(
        table=table,
        chart=(f'<figure><img src="{chart}" alt="edge gap">'
               f"<figcaption>Gross and net expectancy for every concept tested, "
               f"against the level a two-step challenge actually requires. The "
               f"distance is not a tuning gap.</figcaption></figure>" if chart else ""))


TEMPLATE = """<title>The Edge Gap</title>
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
.wrap{{max-width:1080px;margin:0 auto;padding:0 24px 96px}}
.prose{{max-width:68ch}}
h1,h2,h3{{font-family:var(--display);text-wrap:balance;line-height:1.15;margin:0}}
h1{{font-size:clamp(2.1rem,5vw,3.4rem);font-weight:700;letter-spacing:-.022em}}
h2{{font-size:1.6rem;font-weight:600;letter-spacing:-.012em;margin:0 0 .6rem}}
h3{{font-size:1.1rem;font-weight:600}}
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
.box{{border-left:4px solid var(--accent);background:var(--raised);padding:22px 26px;
  border-radius:0 3px 3px 0;margin:0 0 24px;box-shadow:var(--shadow)}}
.box h3{{margin-bottom:.45rem}}
.box p:last-child{{margin-bottom:0}}
.box.hard{{border-left-color:var(--fail)}}
.box.good{{border-left-color:var(--pass)}}
.tablewrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;
  background:var(--raised);box-shadow:var(--shadow)}}
table{{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.78rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  font-weight:600;background:var(--sunk)}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:last-child td{{border-bottom:none}}
.pos{{color:var(--pass)}} .neg{{color:var(--fail)}}
figure{{margin:0 0 26px}}
figure img{{width:100%;height:auto;display:block;border:1px solid var(--rule);
  border-radius:3px;background:#0d1117}}
figcaption{{font-size:.86rem;color:var(--muted);margin-top:9px;max-width:70ch}}
ul.plain{{padding-left:1.1rem}} ul.plain li{{margin-bottom:.6rem}}
hr{{border:none;border-top:1px solid var(--rule);margin:0}}
footer{{padding:44px 0 0;color:var(--muted);font-size:.86rem}}
@media (max-width:640px){{body{{font-size:16px}}.wrap{{padding:0 16px 64px}}
  header.top{{padding:44px 0 30px}}.meta{{gap:18px}}}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">Second attempt &middot; 9 instruments &middot; 2015&ndash;2020 design window</p>
  <h1>The edge gap</h1>
  <p class="dek">I rebuilt the method so the last failure could not repeat, then tested
  five concepts properly. None of them works. The reason is worth more than another
  strategy would have been.</p>
  <div class="meta">
    <div>Concepts tested<b>5</b></div>
    <div>Instruments<b>9</b></div>
    <div>Trades measured<b>10,977</b></div>
    <div>Beating noise<b>0</b></div>
  </div>
</header>

<section>
  <div class="section-head prose">
    <p class="eyebrow">The answer</p>
    <h2>I could not find a better strategy</h2>
  </div>
  <div class="prose">
    <div class="box hard">
      <h3>Not one concept produced an edge distinguishable from zero &mdash; before costs.</h3>
      <p>Gross expectancy came in between +0.038 and +0.062 R per trade, with
      t-statistics of 1.4 to 1.9 on samples of 641 to 4,267 trades. Every one of
      those is inside the noise band. After spread and commission, the best
      survivor earns +0.043 R.</p>
      <p>A two-step challenge needs roughly <b>+0.30 R per trade</b> over 60 to 90
      trades a phase to clear +8% and then +5% without touching a 10% loss limit.
      That is about six times what anything here produced.</p>
    </div>
    <p>I could have handed you a strategy anyway. Tuning five concepts across nine
    markets and twelve years gives enough combinations that something always looks
    excellent somewhere &mdash; which is exactly how the last one was born. The
    honest output of this search is the gap, not a strategy.</p>
  </div>
  {chart}
</section>

<section>
  <div class="section-head">
    <p class="eyebrow">Evidence</p>
    <h2>Every concept, measured the same way</h2>
  </div>
  <div class="tablewrap">{table}</div>
  <div class="prose" style="margin-top:20px">
    <p>Breadth is the share of instrument-year cells with positive expectancy. Around
    50% is what a coin flip produces, and that is where these sit. The old A3 sits at
    43% with the same treatment, which is the control behaving as it should.</p>
    <p>One result deserves a warning label. Short-term reversal looked genuinely
    promising on three instruments &mdash; 67% breadth, +0.06 R, five of six years
    positive. Extending to nine instruments took it to 44% breadth and
    &minus;0.021 R. Three markets was simply too small a sample to tell an edge from
    a run of luck, and that is the same illusion in miniature that produced the
    previous strategy.</p>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Why this matters</p>
    <h2>Costs are not the problem</h2>
  </div>
  <div class="prose">
    <p>This was the diagnostic worth running. If these rules earned +0.15 R gross and
    lost it to spread, the answer would be a cheaper broker or a larger account, and
    there would be something to work with. They do not. Set spread, commission and
    slippage all to zero and the signals still only predict about five hundredths of a
    risk unit per trade, at a significance level that would not survive a referee.</p>
    <p>In other words the rules barely forecast anything. Costs then finish off what
    little there is &mdash; between 0.012 and 0.081 R per trade, with the breakout
    family paying the most because it trades most often.</p>
    <div class="box">
      <h3>The one thread worth pulling</h3>
      <p>Turn-of-month is the most efficient of the five: it keeps +0.043 of its
      +0.055 gross because it trades rarely and holds for days. Its t-statistic of
      1.73 on only 641 trades is the least dead of the set per unit of sample. It is
      still not significant, and 12 trades a year cannot pass a challenge, but if
      anything here deserved more data it is the low-frequency end where costs do not
      dominate.</p>
    </div>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">The method</p>
    <h2>What was done differently</h2>
  </div>
  <div class="prose">
    <p>The protocol was written down before any testing: design on 2015&ndash;2020,
    hold out 2021&ndash;2026, and accept a candidate only on breadth &mdash; positive
    in at least four of six design years, at least 40 trades a year, +0.10 R after
    costs, and working on at least three instruments. Nothing was selected on
    held-out evidence, and the held-out years were never opened, because nothing
    earned the right to be tested there.</p>
    <p>Concepts were chosen for evidence outside this dataset rather than invented:
    time-series momentum, the turn-of-the-month effect, short-term reversal, and
    volatility contraction. The old A3 rode along as a control, and its behaviour
    &mdash; 43% breadth, negative net &mdash; is how I know the screen has teeth.</p>
    <div class="box good">
      <h3>The process caught two things that would have produced another fake</h3>
      <p>A break-even lock written as an absolute 0.10 price units is ten cents of
      gold but a thousand pips of EURUSD. On short trades it placed the stop deep in
      profit, so every stop-out booked a win and short-term reversal reported
      <b>+4.5 R per trade</b> on FX. There is now a regression test that runs the
      same trade at two price scales and fails if they disagree.</p>
      <p>Separately, gold's apparent 22:00 UTC session edge &mdash; +4.5 basis points
      an hour, present in both halves of the design window &mdash; is a spread
      artifact. At broker rollover the bid falls while the mid does not move. On mid
      prices the effect is a third the size and costs more to trade than it pays.</p>
    </div>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Where to go</p>
    <h2>What would actually change the answer</h2>
  </div>
  <div class="prose">
    <p>The gap is a factor of six. That is not closed by tuning; it needs a different
    kind of edge.</p>
    <ul class="plain">
      <li><b>Information these tests cannot see.</b> Everything here is derived from
      past prices on nine liquid markets. Order flow, positioning, earnings and
      macro releases, options surfaces and cross-sectional relative value are all
      outside the dataset, and are where most surviving systematic edges live.</li>
      <li><b>A structurally cheaper cost base.</b> Costs run 0.01&ndash;0.08 R per
      trade. That matters at the margin, though on these numbers it would not be
      enough on its own.</li>
      <li><b>A different objective.</b> +13% inside a 10% loss limit is a demanding
      target. An edge of +0.05 R across many markets is a real if modest business at
      a longer horizon with a bigger account; it simply cannot clear this particular
      hurdle.</li>
      <li><b>Accepting the bet for what it is.</b> If you want to run a challenge
      anyway, treat the fee as the cost of a lottery ticket rather than an
      investment with an expected return. Nothing found here supports a better
      framing than that.</li>
    </ul>
    <p>What did survive is the apparatus: an eleven-year, nine-instrument bid/ask
    dataset, an engine whose costs and prop-firm rules are modelled properly, a
    portfolio simulator for running one account across many markets, and a screen
    that took a strategy which looked excellent and showed it was noise. Any future
    idea can be put through that in minutes. It is the part that would have saved
    the fee.</p>
  </div>
</section>

<footer>
  <hr>
  <p style="margin-top:22px">Dukascopy hourly bid/ask, Jan 2015 &ndash; Jul 2026, nine
  instruments across forex, metals and equity indices. FundedNext Stellar 2-Step $6,000
  rules and cost base. Simulated results from historical data; not a prediction and not
  investment advice.</p>
</footer>
</div>
"""


if __name__ == "__main__":
    html = build()
    out = os.path.join(REPORTS, "report_edge_gap.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
