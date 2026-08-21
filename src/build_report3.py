"""Final go / no-go report: a decade of validation, and the payout mechanics."""

from __future__ import annotations

import base64
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")

ERAS = [
    ("2015-2017", 86, 31, 0, 0, 0, 9.08, 1.92),
    ("2018-2020", 89, 28, 6, 6, 0, 9.88, 1.99),
    ("2021-2022", 52, 33, 8, 8, 0, 9.75, 1.62),
    ("2023-2024", 52, 21, 12, 12, 21, 9.51, 5.28),
    ("2025-2026", 40, 90, 52, 52, 0, 4.25, 1.89),
]

CADENCE = [("21 / 14 days", 1.00, 415), ("60 / 45 days", 1.00, 415),
           ("90 / 60 days", 1.00, 487), ("120 / 90 days", 1.00, 802)]

CONCENTRATION = [("30 days", 5, 15, 9, 1, 93), ("60 days", 24, 48, 40, 6, 49),
                 ("90 days", 36, 51, 63, 13, 35)]


def img(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build():
    ed = pd.read_csv(os.path.join(REPORTS, "decade_edge.csv"))
    pre = ed[ed["year"] <= 2024]

    yrows = []
    for _, r in ed.iterrows():
        built = int(r["year"]) >= 2025
        cls = "built" if built else ("good" if r["expectancy_R"] > 0 else "bad")
        yrows.append(
            f"<tr class='{cls}'><td>{int(r['year'])}{' &#9679;' if built else ''}</td>"
            f"<td class='n'>{int(r['trades'])}</td>"
            f"<td class='n'>{r['win_rate'] * 100:.1f}</td>"
            f"<td class='n {'pos' if r['expectancy_R'] > 0 else 'neg'}'>"
            f"{r['expectancy_R']:+.3f}</td>"
            f"<td class='n {'pos' if r['total_R'] > 0 else 'neg'}'>{r['total_R']:+.1f}</td>"
            f"<td class='n'>{r['pf']:.2f}</td>"
            f"<td class='n'>{r['max_dd_pct']:.1f}</td></tr>")
    year_table = (
        "<table><thead><tr><th>Year</th><th class='n'>Trades</th>"
        "<th class='n'>Win %</th><th class='n'>Exp R</th><th class='n'>Total R</th>"
        f"<th class='n'>PF</th><th class='n'>Max DD %</th></tr></thead>"
        f"<tbody>{''.join(yrows)}</tbody></table>")

    erows = []
    for era, runs, p1, fund, alive, breach, dd, wd in ERAS:
        built = era == "2025-2026"
        erows.append(
            f"<tr class='{'built' if built else ''}'><td>{esc(era)}</td>"
            f"<td class='n'>{runs}</td>"
            f"<td class='n {'pos' if p1 > 60 else 'neg'}'>{p1}</td>"
            f"<td class='n {'pos' if fund > 40 else 'neg'}'>{fund}</td>"
            f"<td class='n'>{alive}</td>"
            f"<td class='n {'neg' if breach > 0 else ''}'>{breach}</td>"
            f"<td class='n'>{dd:.1f}</td><td class='n'>{wd:.2f}</td></tr>")
    era_table = (
        "<table><thead><tr><th>Era</th><th class='n'>Runs</th>"
        "<th class='n'>Phase 1 %</th><th class='n'>Funded %</th>"
        "<th class='n'>Alive %</th><th class='n'>Breached %</th>"
        f"<th class='n'>DD med</th><th class='n'>Worst day</th></tr></thead>"
        f"<tbody>{''.join(erows)}</tbody></table>")

    crows = "".join(
        f"<tr><td>{esc(w)}</td><td class='n'>{a1}</td><td class='n'>{a2}</td>"
        f"<td class='n {'pos' if a3 >= 50 else ''}'>{a3}</td><td class='n'>{a4}</td>"
        f"<td class='n {'pos' if share <= 40 else 'neg'}'>{share}%</td></tr>"
        for w, a1, a2, a3, a4, share in CONCENTRATION)
    conc_table = (
        "<table><thead><tr><th>Wait before withdrawing</th>"
        "<th class='n'>A1 %</th><th class='n'>A2 %</th><th class='n'>A3 %</th>"
        "<th class='n'>A4 %</th><th class='n'>A3 best-day share</th></tr></thead>"
        f"<tbody>{crows}</tbody></table>")

    prows = "".join(
        f"<tr><td>{esc(c)}</td><td class='n'>{n:.2f}</td>"
        f"<td class='n'>${p}</td><td class='n neg'>0</td></tr>"
        for c, n, p in CADENCE)
    pay_table = (
        "<table><thead><tr><th>Withdrawal cadence</th>"
        "<th class='n'>Payouts per run</th><th class='n'>Amount</th>"
        f"<th class='n'>Still alive %</th></tr></thead><tbody>{prows}</tbody></table>")

    chart = img(os.path.join(REPORTS, "decade_edge.png"))

    return TEMPLATE.format(
        year_table=year_table, era_table=era_table,
        conc_table=conc_table, pay_table=pay_table,
        chart=(f'<figure><img src="{chart}" alt="expectancy by year">'
               f"<figcaption>Every year 2015-2026, same code and parameters. The "
               f"cumulative line spends the whole decade underwater and only climbs "
               f"back toward zero inside the window the strategy was built in."
               f"</figcaption></figure>" if chart else ""),
        pre_years=len(pre),
        pre_pos=int((pre["expectancy_R"] > 0).sum()),
        pre_mean=f"{pre['expectancy_R'].mean():+.3f}",
        pre_total=f"{pre['total_R'].sum():+.1f}",
        pre_trades=int(pre["trades"].sum()),
    )


TEMPLATE = """<title>The Decade Test</title>
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
h3{{font-size:1.08rem;font-weight:600}}
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
.verdict-box{{border-left:4px solid var(--fail);background:var(--raised);
  padding:24px 28px;border-radius:0 3px 3px 0;margin:0 0 26px;box-shadow:var(--shadow)}}
.verdict-box h3{{margin-bottom:.5rem;font-size:1.3rem;color:var(--fail)}}
.verdict-box p:last-child{{margin-bottom:0}}
.note-box{{border-left:3px solid var(--accent);background:var(--raised);
  padding:18px 22px;border-radius:0 3px 3px 0;margin:0 0 20px;box-shadow:var(--shadow)}}
.note-box h3{{margin-bottom:.4rem}}
.note-box p:last-child{{margin-bottom:0}}
.tablewrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;
  background:var(--raised);box-shadow:var(--shadow)}}
table{{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.78rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule);white-space:nowrap}}
thead th{{font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);
  font-weight:600;background:var(--sunk);position:sticky;top:0}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:last-child td{{border-bottom:none}}
tr.built td{{background:color-mix(in srgb,var(--accent) 12%,transparent);font-weight:600}}
tr.bad td:first-child{{border-left:2px solid var(--fail)}}
tr.good td:first-child{{border-left:2px solid var(--pass)}}
.pos{{color:var(--pass)}} .neg{{color:var(--fail)}}
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
  <p class="eyebrow">Go / no-go &middot; A3 Donchian H4 &middot; XAUUSD 2015&ndash;2026</p>
  <h1>The decade test</h1>
  <p class="dek">Before spending $59.99, I ran the strategy across eleven years of
  gold instead of the twenty months it was built in. It loses money in nine of
  them.</p>
  <div class="meta">
    <div>Years tested<b>12</b></div>
    <div>Trades 2015&ndash;2024<b>{pre_trades}</b></div>
    <div>Result 2015&ndash;2024<b>{pre_total} R</b></div>
    <div>Verdict<b>Do not trade</b></div>
  </div>
</header>

<section>
  <div class="section-head prose">
    <p class="eyebrow">The answer</p>
    <h2>Do not fund this account</h2>
  </div>
  <div class="prose">
    <div class="verdict-box">
      <h3>The edge belongs to 2025&ndash;26, not to the strategy.</h3>
      <p>Across {pre_years} years from 2015 to 2024 &mdash; {pre_trades} trades,
      same code, same parameters, no refitting &mdash; A3 returns
      <b>{pre_mean} R per trade</b> and <b>{pre_total} R in total</b>. It is
      profitable in only {pre_pos} of those {pre_years} years, and never by much.
      The two clearly good years are the two it was built and tested in.</p>
      <p>That is the definition of a strategy fitted to its sample. The 88%
      funded rate I reported earlier was real for that window and means nothing
      about the next one.</p>
    </div>
  </div>
  {chart}
</section>

<section>
  <div class="section-head">
    <p class="eyebrow">Evidence</p>
    <h2>Year by year</h2>
  </div>
  <div class="tablewrap">{year_table}</div>
  <div class="prose" style="margin-top:20px">
    <p class="muted">Marked years are the development and test windows from the
    earlier reports. 2026 covers January to July only.</p>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Evidence</p>
    <h2>What a challenge would have done</h2>
  </div>
  <div class="prose">
    <p>The same picture in the currency that matters: simulated challenges started
    every ten days, each given 205 days to pass both phases and trade funded.</p>
  </div>
  <div class="tablewrap">{era_table}</div>
  <div class="prose" style="margin-top:22px">
    <p>Outside 2025&ndash;26 the account clears Phase 1 in roughly a fifth to a
    third of attempts and reaches funded almost never. In 2023&ndash;24 it also
    <b>breached outright in 21%</b> of starts, with a worst day of 5.28% against
    a 5% hard limit &mdash; that is an account already dead.</p>
    <div class="note-box">
      <h3>And these numbers are the generous version</h3>
      <p>The decade test runs on hourly bars, which flatter this strategy by
      roughly 1.5&ndash;2&times;: on the overlapping period the identical code
      reports +0.620 R per trade on hourly data against +0.264 on minute data,
      because the trailing stop ratchets once an hour instead of once a minute.
      The real pre-2025 record is worse than shown.</p>
    </div>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Separately</p>
    <h2>The payout mechanics, which I had wrong</h2>
  </div>
  <div class="prose">
    <p>Two FundedNext rules were missing from the original model: a reward request
    needs 2% account growth, and a 40% consistency rule caps the best single day at
    40% of the profit withdrawn. The first costs nothing. The second bites hard,
    because withdrawing at the 21-day minimum means the profit sits in one or two
    trades.</p>
    <p>Waiting fixes the consistency problem. This is the share of profitable
    windows that would clear the 40% rule, by how long you wait:</p>
  </div>
  <div class="tablewrap">{conc_table}</div>
  <div class="prose" style="margin-top:22px">
    <p>And in full simulations with a funded account held long enough to matter,
    patience roughly doubles the payout &mdash; but the account does not survive
    either way:</p>
  </div>
  <div class="tablewrap">{pay_table}</div>
  <div class="prose" style="margin-top:22px">
    <div class="note-box">
      <h3>One payout, then the account dies</h3>
      <p>Every funded account in the long-horizon runs eventually breached. That is
      structural rather than bad luck: the drawdown floor is <b>static</b> at $5,400,
      and withdrawing resets the balance to $6,000, so the account never builds a
      buffer. It is on a permanent 10% leash. Expect roughly one reward per funded
      account, then another challenge fee.</p>
    </div>
    <p>None of this matters much now, because it only applies once you have an edge
    worth withdrawing. It is recorded so the mechanics are understood if a future
    strategy earns the chance.</p>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Where this leaves things</p>
    <h2>What I would do</h2>
  </div>
  <div class="prose">
    <ul class="plain">
      <li><b>Do not buy a challenge for this strategy.</b> The $59.99 is not the
      real cost; the real cost is a few months spent trading something with no
      demonstrated edge.</li>
      <li><b>Keep the harness, drop the strategy.</b> The engine, the eleven-year
      dataset and this validation loop are the durable part of the work. Any future
      idea can be run through the decade test in minutes, and that test is the thing
      that would have saved the fee.</li>
      <li><b>Raise the bar for what counts as a candidate.</b> Positive expectancy
      in the window it was designed in means nothing. The requirement should be
      profitable across most of 2015&ndash;2024, on minute data, before any money
      is spent.</li>
      <li><b>The spec and the EA are still worth having.</b> Both are written and
      match the backtest. If you want to watch the strategy behave on a demo
      account to sanity-check fills and spreads, they are ready &mdash; just do not
      point them at a funded account.</li>
    </ul>
  </div>
</section>

<footer>
  <hr>
  <p style="margin-top:22px">Dukascopy XAUUSD hourly bid/ask, Jan 2015 &ndash; Jul 2026,
  68,570 bars. FundedNext Stellar 2-Step $6,000 rules with contract size 100,
  0.0016% notional commission and spread taken from the feed. Simulated results from
  historical data; not a prediction and not investment advice.</p>
</footer>
</div>
"""


if __name__ == "__main__":
    html = build()
    out = os.path.join(REPORTS, "report_decade.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
