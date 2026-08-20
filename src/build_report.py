"""Assemble the self-contained HTML report from the test-run artefacts.

Charts are embedded as data URIs so the published page has no external
dependencies beyond the Google Fonts stylesheet.
"""

from __future__ import annotations

import base64
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from concept_log import CONCEPTS, SUMMARY  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")

STRATEGY_KEYS = [
    ("A1_DonchianH1", "A1 Donchian H1 Breakout"),
    ("A2_TrendPullback", "A2 H1 Trend Pullback"),
    ("A3_DonchianH4", "A3 Donchian H4 Swing"),
    ("A4_AlignedMomentum", "A4 Multi-Timeframe Aligned Momentum"),
]

BLURBS = {
    "A1 Donchian H1 Breakout": (
        "Buys a break of the highest high of the last 45 hourly bars and sells the "
        "mirror image, whenever ADX confirms the market is actually moving. The "
        "channel defines its own trend, so no separate trend filter is applied."),
    "A2 H1 Trend Pullback": (
        "Waits for an established H1 and H4 uptrend, lets price pull back into the "
        "H1 20-EMA, then enters when an hourly bar reasserts the trend. The stop "
        "sits behind structure rather than at an arbitrary distance."),
    "A3 Donchian H4 Swing": (
        "The channel idea on a four-hour clock: a break of the 20-bar H4 extreme "
        "with a stop of one H4 ATR. Fewer, longer trades that hold through noise "
        "the hourly version gets shaken out by."),
    "A4 Multi-Timeframe Aligned Momentum": (
        "Only engages when the hourly, four-hour and daily trends all agree, then "
        "buys an 8-bar breakout in that direction. Flat most of the time by design."),
}

SPECS = {
    "A1 Donchian H1 Breakout": [
        ("Signal", "45-bar H1 Donchian break"), ("Stop", "2.5 x H1 ATR"),
        ("Target", "5R, trailing from 2R"), ("Filter", "ADX(14) > 18"),
        ("Risk", "1.00% per trade"), ("Cap", "2 trades / 2 losses per day"),
    ],
    "A2 H1 Trend Pullback": [
        ("Signal", "Pullback to H1 EMA20, then H1 continuation"),
        ("Stop", "2.0 x H1 ATR"), ("Target", "2.5R, trailing from 2R"),
        ("Filter", "H1+H4 trend, ADX(14) > 20"), ("Risk", "0.75% per trade"),
        ("Cap", "2 trades / 2 losses per day"),
    ],
    "A3 Donchian H4 Swing": [
        ("Signal", "20-bar H4 Donchian break"), ("Stop", "1.0 x H4 ATR"),
        ("Target", "4R, trailing from 2R"), ("Filter", "ADX(14) > 15 on H4"),
        ("Risk", "0.75% per trade"), ("Cap", "1 trade / 1 loss per day"),
    ],
    "A4 Multi-Timeframe Aligned Momentum": [
        ("Signal", "8-bar H1 break with H1+H4+D1 aligned"),
        ("Stop", "2.5 x H1 ATR"), ("Target", "4R, trailing from 2R"),
        ("Filter", "ADX(14) > 25"), ("Risk", "0.75% per trade"),
        ("Cap", "2 trades / 2 losses per day"),
    ],
}


def img(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


def esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def stage_label(outcome: str, detail: str) -> tuple[str, str]:
    if outcome == "PASS":
        return "pass", "passed"
    if outcome == "BREACH":
        return "fail", f"breached &mdash; {esc(detail)}"
    return "open", "unfinished at data end"


def build() -> str:
    with open(os.path.join(REPORTS, "test_summary.json")) as fh:
        summary = json.load(fh)
    tbl = pd.read_csv(os.path.join(REPORTS, "test_stage_results.csv"))
    corr = pd.read_csv(os.path.join(REPORTS, "correlation.csv"), index_col=0)
    monthly = pd.read_csv(os.path.join(REPORTS, "strategy_monthly.csv"), index_col=0)
    regime = pd.read_csv(os.path.join(REPORTS, "regime_monthly.csv"), index_col=0)

    # ---------- verdict cards -------------------------------------------
    cards = []
    for key, name in STRATEGY_KEYS:
        s = summary.get(name)
        if not s:
            continue
        stages = {r["stage"]: r for r in s["stages"]}
        funded = s["reached_funded"]
        pay = sum(st.get("payout_net_usd", 0) for st in s["stages"])
        final = s["stages"][-1]["final_balance"]
        breached = any(st["outcome"] == "BREACH" for st in s["stages"])

        if funded and pay > 0:
            tone, verdict = "pass", "Funded &amp; paid out"
        elif funded:
            tone, verdict = "warn", "Funded, no payout"
        elif breached:
            tone, verdict = "fail", "Account breached"
        else:
            tone, verdict = "open", "Did not finish"

        rows = []
        for stg in ("phase1", "phase2", "funded"):
            r = stages.get(stg)
            label = {"phase1": "Phase 1", "phase2": "Phase 2", "funded": "Funded"}[stg]
            if not r:
                rows.append(f'<li><span class="k">{label}</span>'
                            f'<span class="v muted">not reached</span></li>')
                continue
            cls, txt = stage_label(r["outcome"], r["detail"])
            rows.append(
                f'<li><span class="k">{label}</span>'
                f'<span class="v"><span class="dot {cls}"></span>{txt}'
                f' <span class="muted">&middot; {r["trades"]} trades &middot; '
                f'PF {r["profit_factor"]:.2f}</span></span></li>')

        cards.append(f"""
        <article class="card {tone}">
          <header>
            <span class="tag">{esc(name.split()[0])}</span>
            <h3>{esc(" ".join(name.split()[1:]))}</h3>
          </header>
          <p class="verdict">{verdict}</p>
          <ul class="stagelist">{"".join(rows)}</ul>
          <dl class="kpis">
            <div><dt>Final balance</dt><dd>${final:,.0f}</dd></div>
            <div><dt>Paid out</dt><dd>${pay:,.0f}</dd></div>
            <div><dt>Attempts</dt><dd>{s.get("attempts", 1)}</dd></div>
          </dl>
        </article>""")

    # ---------- stage results table --------------------------------------
    head = ("<tr><th>Account</th><th>Stage</th><th>Outcome</th><th class='n'>Days</th>"
            "<th class='n'>Trades</th><th class='n'>Win %</th><th class='n'>PF</th>"
            "<th class='n'>Net $</th><th class='n'>Max DD %</th>"
            "<th class='n'>Worst day %</th></tr>")
    body = []
    for _, r in tbl.iterrows():
        cls, _ = stage_label(r["outcome"], r["detail"])
        net = r["net_usd"]
        body.append(
            f"<tr><td>{esc(r['strategy'].split()[0])}</td><td>{esc(r['stage'])}</td>"
            f"<td><span class='dot {cls}'></span>{esc(r['outcome'])}</td>"
            f"<td class='n'>{r['calendar_days']:.0f}</td><td class='n'>{r['trades']}</td>"
            f"<td class='n'>{r['win_rate_pct']:.1f}</td>"
            f"<td class='n'>{r['profit_factor']:.2f}</td>"
            f"<td class='n {'neg' if net < 0 else 'pos'}'>{net:,.0f}</td>"
            f"<td class='n'>{r['max_dd_pct']:.2f}</td>"
            f"<td class='n'>{r['worst_day_pct']:.2f}</td></tr>")
    stage_table = f"<table><thead>{head}</thead><tbody>{''.join(body)}</tbody></table>"

    # ---------- monthly expectancy heat table ----------------------------
    cols = [c for c in monthly.columns if c != "n_pos"]
    mhead = "<tr><th>Month</th>" + "".join(f"<th class='n'>{esc(c)}</th>" for c in cols)
    mhead += "<th class='n'>ATR %</th><th class='n'>Return %</th></tr>"
    mrows = []
    for month in monthly.index:
        cells = []
        for c in cols:
            v = monthly.loc[month, c]
            if pd.isna(v):
                cells.append("<td class='n muted'>&ndash;</td>")
            else:
                tone = "pos" if v > 0 else "neg"
                strength = min(abs(v) / 1.2, 1.0)
                cells.append(f"<td class='n {tone}' style='--w:{strength:.2f}'>{v:+.2f}</td>")
        atr = regime.loc[month, "atr_pct"] if month in regime.index else float("nan")
        ret = regime.loc[month, "ret_pct"] if month in regime.index else float("nan")
        era = "test" if str(month) >= "2025-12" else "dev"
        mrows.append(
            f"<tr class='{era}'><td>{esc(month)}</td>{''.join(cells)}"
            f"<td class='n muted'>{atr:.2f}</td>"
            f"<td class='n muted'>{ret:+.1f}</td></tr>")
    monthly_table = (f"<table class='heat'><thead>{mhead}</thead>"
                     f"<tbody>{''.join(mrows)}</tbody></table>")

    # ---------- correlation ----------------------------------------------
    short = [c.split()[0] for c in corr.columns]
    chead = "<tr><th></th>" + "".join(f"<th class='n'>{esc(s)}</th>" for s in short) + "</tr>"
    crows = []
    for i, idx in enumerate(corr.index):
        cells = []
        for j, c in enumerate(corr.columns):
            v = corr.iloc[i, j]
            if i == j:
                cells.append("<td class='n muted'>&mdash;</td>")
            else:
                cells.append(f"<td class='n' style='--w:{abs(v):.2f}'>{v:.2f}</td>")
        crows.append(f"<tr><td>{esc(short[i])}</td>{''.join(cells)}</tr>")
    corr_table = (f"<table class='heat corr'><thead>{chead}</thead>"
                  f"<tbody>{''.join(crows)}</tbody></table>")

    # ---------- concept log ----------------------------------------------
    vmap = {"promoted": "pass", "rejected": "fail", "infeasible": "fail",
            "reworked": "warn"}
    lrows = []
    for name, fam, n, e, pf, verdict, note in CONCEPTS:
        lrows.append(
            f"<tr><td>{esc(name)}</td><td class='muted'>{esc(fam)}</td>"
            f"<td class='n'>{n}</td>"
            f"<td class='n {'pos' if e > 0 else 'neg'}'>{e:+.3f}</td>"
            f"<td class='n'>{pf:.2f}</td>"
            f"<td><span class='dot {vmap[verdict]}'></span>{esc(verdict)}</td>"
            f"<td class='note'>{esc(note)}</td></tr>")
    concept_table = (
        "<table><thead><tr><th>Concept</th><th>Family</th><th class='n'>Trades</th>"
        "<th class='n'>Exp. R</th><th class='n'>PF</th><th>Verdict</th><th>Why</th>"
        f"</tr></thead><tbody>{''.join(lrows)}</tbody></table>")

    # ---------- strategy cards -------------------------------------------
    scards = []
    for key, name in STRATEGY_KEYS:
        specs = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>"
                        for k, v in SPECS[name])
        scards.append(f"""
        <article class="strategy">
          <h3><span class="tag">{esc(name.split()[0])}</span>
              {esc(" ".join(name.split()[1:]))}</h3>
          <p>{BLURBS[name]}</p>
          <dl class="spec">{specs}</dl>
        </article>""")

    # ---------- charts ----------------------------------------------------
    figs = []
    for key, name in STRATEGY_KEYS:
        d = img(os.path.join(REPORTS, f"{key}_journey.png"))
        if d:
            figs.append(f"""<figure><img src="{d}" alt="{esc(name)} equity curve">
              <figcaption>{esc(name)} &mdash; the full journey. Dotted lines are the
              two profit targets, the dashed red line is the $5,400 static floor that
              ends the account.</figcaption></figure>""")
    funded_figs = []
    for key, name in STRATEGY_KEYS:
        d = img(os.path.join(REPORTS, f"{key}_funded.png"))
        if d:
            funded_figs.append(f"""<figure><img src="{d}" alt="{esc(name)} funded">
              <figcaption>{esc(name)} &mdash; life after passing. This is the part that
              actually pays.</figcaption></figure>""")
    portfolio = img(os.path.join(REPORTS, "portfolio.png"))

    avg_corr = summary.get("_avg_pairwise_correlation", 0.0)
    total_paid = sum(sum(st.get("payout_net_usd", 0) for st in s["stages"])
                     for k, s in summary.items() if isinstance(s, dict) and "stages" in s)
    n_funded = sum(1 for k, s in summary.items()
                   if isinstance(s, dict) and s.get("reached_funded"))
    n_breach = sum(1 for k, s in summary.items() if isinstance(s, dict) and "stages" in s
                   and any(st["outcome"] == "BREACH" for st in s["stages"]))

    return TEMPLATE.format(
        cards="".join(cards),
        stage_table=stage_table,
        monthly_table=monthly_table,
        corr_table=corr_table,
        concept_table=concept_table,
        strategy_cards="".join(scards),
        figures="".join(figs),
        funded_figures="".join(funded_figs) or
        "<p class='muted'>No account reached the funded stage with a payout to chart.</p>",
        portfolio=(f'<figure><img src="{portfolio}" alt="all four accounts">'
                   f"<figcaption>All four accounts, same start date, same market."
                   f"</figcaption></figure>" if portfolio else ""),
        avg_corr=f"{avg_corr:.2f}",
        total_paid=f"{total_paid:,.0f}",
        n_funded=n_funded,
        n_breach=n_breach,
        n_tested=SUMMARY["tested"],
        n_rejected=SUMMARY["rejected"],
    )


TEMPLATE = """<title>Four Gold Accounts</title>
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
body{{
  margin:0; background:var(--paper); color:var(--ink);
  font-family:var(--body); font-size:17px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 24px 96px}}
.prose{{max-width:68ch}}
h1,h2,h3,h4{{font-family:var(--display);text-wrap:balance;line-height:1.15;margin:0}}
h1{{font-size:clamp(2.1rem,5vw,3.4rem);font-weight:700;letter-spacing:-.022em}}
h2{{font-size:1.6rem;font-weight:600;letter-spacing:-.012em;margin:0 0 .6rem}}
h3{{font-size:1.06rem;font-weight:600;letter-spacing:-.006em}}
p{{margin:0 0 1.05rem}}
a{{color:var(--accent)}}
.muted{{color:var(--muted)}}
.eyebrow{{
  font-family:var(--mono);font-size:.72rem;font-weight:500;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 1rem
}}
header.top{{padding:72px 0 40px;border-bottom:1px solid var(--rule)}}
.dek{{font-size:1.22rem;color:var(--muted);max-width:60ch;margin-top:1.1rem}}
.meta{{
  display:flex;flex-wrap:wrap;gap:28px;margin-top:32px;
  font-family:var(--mono);font-size:.78rem;color:var(--muted)
}}
.meta b{{display:block;color:var(--ink);font-size:1.32rem;font-weight:600;
  letter-spacing:-.01em;font-variant-numeric:tabular-nums}}
section{{padding:56px 0 0}}
.section-head{{margin-bottom:26px}}
.section-head .eyebrow{{margin-bottom:.5rem}}

/* verdict cards */
.cards{{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(238px,1fr))}}
.card{{
  background:var(--raised);border:1px solid var(--rule);border-radius:3px;
  padding:20px;box-shadow:var(--shadow);border-top:3px solid var(--muted)
}}
.card.pass{{border-top-color:var(--pass)}}
.card.warn{{border-top-color:var(--warn)}}
.card.fail{{border-top-color:var(--fail)}}
.card.open{{border-top-color:var(--muted)}}
.card header{{display:flex;align-items:baseline;gap:9px;margin-bottom:10px}}
.tag{{
  font-family:var(--mono);font-size:.7rem;font-weight:600;letter-spacing:.06em;
  color:var(--accent);border:1px solid var(--rule);border-radius:2px;padding:1px 6px
}}
.card h3{{font-size:.95rem}}
.verdict{{font-family:var(--display);font-weight:600;font-size:1.08rem;margin:0 0 12px}}
.card.pass .verdict{{color:var(--pass)}}
.card.warn .verdict{{color:var(--warn)}}
.card.fail .verdict{{color:var(--fail)}}
.stagelist{{list-style:none;margin:0 0 14px;padding:0;
  border-top:1px solid var(--rule)}}
.stagelist li{{display:flex;gap:10px;justify-content:space-between;
  padding:7px 0;border-bottom:1px solid var(--rule);font-size:.82rem}}
.stagelist .k{{font-family:var(--mono);font-size:.72rem;color:var(--muted);
  text-transform:uppercase;letter-spacing:.07em;white-space:nowrap;padding-top:2px}}
.stagelist .v{{text-align:right}}
.dot{{display:inline-block;width:7px;height:7px;border-radius:50%;
  margin-right:6px;background:var(--muted);vertical-align:middle}}
.dot.pass{{background:var(--pass)}} .dot.fail{{background:var(--fail)}}
.dot.warn{{background:var(--warn)}} .dot.open{{background:var(--muted)}}
.kpis{{display:grid;grid-template-columns:1.2fr 1fr .7fr;gap:10px;margin:0}}
.kpis dt{{font-family:var(--mono);font-size:.62rem;letter-spacing:.06em;
  text-transform:uppercase;color:var(--muted);white-space:nowrap}}
.kpis dd{{margin:2px 0 0;font-family:var(--mono);font-size:.94rem;font-weight:600;
  font-variant-numeric:tabular-nums}}

/* tables */
.tablewrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:3px;
  background:var(--raised);box-shadow:var(--shadow)}}
table{{border-collapse:collapse;width:100%;font-family:var(--mono);font-size:.79rem}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid var(--rule);
  white-space:nowrap}}
thead th{{
  font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);font-weight:600;background:var(--sunk);position:sticky;top:0
}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}}
tbody tr:last-child td{{border-bottom:none}}
.pos{{color:var(--pass)}} .neg{{color:var(--fail)}}
td.note{{white-space:normal;min-width:32ch;font-family:var(--body);
  font-size:.86rem;color:var(--muted)}}
.heat td.n[style]{{position:relative}}
.heat td.n[style]::before{{
  content:"";position:absolute;inset:2px;border-radius:2px;
  background:currentColor;opacity:calc(var(--w,0)*.14);z-index:0
}}
.heat tr.test td:first-child{{border-left:2px solid var(--accent)}}
.corr td.n[style]::before{{background:var(--accent)}}

/* strategies */
.strategies{{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(310px,1fr))}}
.strategy{{background:var(--raised);border:1px solid var(--rule);border-radius:3px;
  padding:22px;box-shadow:var(--shadow)}}
.strategy h3{{display:flex;align-items:baseline;gap:9px;margin-bottom:.7rem}}
.strategy p{{font-size:.94rem;margin-bottom:1rem}}
.spec{{margin:0;display:grid;gap:0;border-top:1px solid var(--rule)}}
.spec div{{display:flex;justify-content:space-between;gap:14px;padding:6px 0;
  border-bottom:1px solid var(--rule)}}
.spec div:last-child{{border-bottom:none}}
.spec dt{{font-family:var(--mono);font-size:.7rem;text-transform:uppercase;
  letter-spacing:.07em;color:var(--muted)}}
.spec dd{{margin:0;font-family:var(--mono);font-size:.76rem;text-align:right}}

/* figures */
figure{{margin:0 0 26px}}
figure img{{width:100%;height:auto;display:block;border:1px solid var(--rule);
  border-radius:3px;background:#0d1117}}
figcaption{{font-size:.86rem;color:var(--muted);margin-top:9px;max-width:70ch}}

/* callouts */
.callout{{
  border-left:3px solid var(--accent);background:var(--raised);
  padding:18px 22px;border-radius:0 3px 3px 0;margin:0 0 22px;
  box-shadow:var(--shadow)
}}
.callout h3{{margin-bottom:.45rem}}
.callout p:last-child{{margin-bottom:0}}
.callout.hard{{border-left-color:var(--fail)}}
ul.plain{{padding-left:1.1rem}}
ul.plain li{{margin-bottom:.55rem}}
hr{{border:none;border-top:1px solid var(--rule);margin:0}}
footer{{padding:44px 0 0;color:var(--muted);font-size:.86rem}}
@media (max-width:640px){{
  body{{font-size:16px}} .wrap{{padding:0 16px 64px}}
  header.top{{padding:44px 0 30px}} .meta{{gap:18px}}
}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
</style>

<div class="wrap">
<header class="top">
  <p class="eyebrow">XAUUSD &middot; Stellar 2-Step $6K &middot; Dec 2025 &ndash; Aug 2026</p>
  <h1>Four gold accounts, eight months, one hostile market</h1>
  <p class="dek">Four strategies built and tuned on 2025 data, then run through the
  Dec 2025 &ndash; Aug 2026 window on real bid/ask data with spread, commission and
  slippage charged on every trade. None of them blew up. One of them paid.</p>
  <div class="meta">
    <div>Concepts tested<b>{n_tested}</b></div>
    <div>Discarded<b>{n_rejected}</b></div>
    <div>Reached funded<b>{n_funded} of 4</b></div>
    <div>Accounts breached<b>{n_breach}</b></div>
    <div>Total paid out<b>${total_paid}</b></div>
  </div>
</header>

<section>
  <div class="section-head">
    <p class="eyebrow">The verdict</p>
    <h2>What each account did</h2>
  </div>
  <div class="cards">{cards}</div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Read this first</p>
    <h2>The honest summary</h2>
  </div>
  <div class="prose">
    <div class="callout hard">
      <h3>These strategies survived the rules. They did not make money.</h3>
      <p>Every account stayed inside the 5% daily and 10% static loss limits for the
      whole eight months &mdash; no breaches, no blown accounts. But the funded stages
      landed in April&ndash;August 2026, and in that stretch nothing worked.</p>
    </div>
    <p>The strategies were built and tuned only on 2025 data, then run once through
    the test window. That separation is the reason these numbers are worth anything,
    and it is also why they are disappointing: gold in 2026 did not behave like gold
    in 2025.</p>
    <p>The single most useful finding here is not a strategy. It is that during
    April&ndash;August 2026 every concept tested was unprofitable &mdash; trend
    following and mean reversion alike. That is a property of the market in those
    months, not a failure to find the right indicator, and no amount of further
    searching would have fixed it without simply fitting to that chop.</p>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Diagnosis</p>
    <h2>Why the edge disappeared</h2>
  </div>
  <div class="prose">
    <p>Gold's daily ATR roughly doubled between the two periods, from 1.44% to 2.38%
    of price, while monthly moves flipped from a steady climb to violent two-way
    swings: +12.6%, +11.3%, &minus;12.8%, &minus;10.7%, +10.7%. The table below is
    each strategy's expectancy per trade, month by month. Rows from December 2025
    onward are the test window.</p>
  </div>
  <div class="tablewrap">{monthly_table}</div>
  <div class="prose" style="margin-top:22px">
    <p>Read down the columns and the pattern is obvious. Through 2025 the strategies
    are positive in most months. From April 2026 they go red together, recover in
    June, go red again in July. The funded accounts opened directly into that.</p>
  </div>
</section>

<section>
  <div class="section-head">
    <p class="eyebrow">Results</p>
    <h2>Stage by stage</h2>
  </div>
  <div class="tablewrap">{stage_table}</div>
</section>

<section>
  <div class="section-head">
    <p class="eyebrow">Equity</p>
    <h2>The full journey</h2>
  </div>
  {figures}
</section>

<section>
  <div class="section-head">
    <p class="eyebrow">After the pass</p>
    <h2>Life as a funded account</h2>
  </div>
  {funded_figures}
</section>

<section>
  <div class="section-head">
    <p class="eyebrow">Together</p>
    <h2>All four at once</h2>
  </div>
  {portfolio}
  <div class="prose" style="margin-top:8px">
    <p>Running four accounts only helps if they can fail independently. These cannot.
    Average pairwise correlation of daily P&amp;L is <b>{avg_corr}</b>, and the two that
    got funded &mdash; A3 and A4 &mdash; are the most correlated pair of the lot, because
    every surviving strategy is a trend-continuation strategy on the same instrument.
    Four accounts here is closer to one position at four times the size than to a
    diversified book.</p>
  </div>
  <div class="tablewrap" style="max-width:520px">{corr_table}</div>
</section>

<section>
  <div class="section-head">
    <p class="eyebrow">The four</p>
    <h2>What each strategy actually does</h2>
  </div>
  <div class="strategies">{strategy_cards}</div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">The search</p>
    <h2>Everything that was tried and thrown away</h2>
  </div>
  <div class="prose">
    <p>{n_tested} concepts were tested on 2025 data and {n_rejected} were discarded
    outright; one more was losing until a wider stop rescued it. Every counter-trend
    idea lost money at every parameter setting tried, which is why all four survivors
    are trend strategies &mdash; not because the set was chosen that way.</p>
  </div>
  <div class="tablewrap">{concept_table}</div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Method</p>
    <h2>How this was tested</h2>
  </div>
  <div class="prose">
    <ol class="plain">
      <li><b>Real bid and ask.</b> Dukascopy M1 candles for both sides of the book,
      Jan 2025 &ndash; Aug 2026. Longs fill at the ask and exit at the bid, shorts the
      reverse, so the spread is measured from the feed rather than assumed. Median
      spread was $0.62.</li>
      <li><b>Dead minutes removed.</b> 21% of the raw feed is zero-volume padding for
      hours when gold is shut. Left in, it decays a weekend ATR toward zero and every
      ATR-scaled stop comes out far too tight.</li>
      <li><b>Costs charged.</b> $7 per lot round-turn commission, $0.05 slippage on
      entries, $0.15 against every stop-out. A bar that covers both the stop and the
      target is scored as a stop.</li>
      <li><b>No lookahead.</b> Signals computed on bar <i>t</i> execute at bar
      <i>t+1</i>'s open.</li>
      <li><b>Rules enforced on equity, not balance.</b> The 5% daily and 10% static
      limits are checked against intrabar equity including open P&amp;L, which is how
      accounts actually die.</li>
      <li><b>Built on 2025, tested once on the rest.</b> Parameters were chosen by
      sweeping 2025 data and picking the middle of broad plateaus, never single best
      cells.</li>
    </ol>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">Caveats</p>
    <h2>What would make me distrust this</h2>
  </div>
  <div class="prose">
    <div class="callout">
      <h3>One market, one period</h3>
      <p>Eight months of one instrument is a handful of independent trends, not a
      statistical sample. The development window was a single strong uptrend, which
      is the easiest possible regime for the strategies that won.</p>
    </div>
    <div class="callout">
      <h3>The test window was consulted</h3>
      <p>The strategies were fixed before the test ran, but after seeing the result I
      went back and changed two things in the engine: how a position is sized when
      the minimum lot is too large, and the order of the stop and breach checks
      within a bar. Both are corrections rather than tuning, and both were re-checked
      on 2025 data &mdash; but the final numbers are no longer purely out of sample.</p>
    </div>
    <div class="callout">
      <h3>A $6,000 account is small for gold</h3>
      <p>At $4,000&ndash;5,000 an ounce with 2&ndash;3% daily ranges, a sensible
      ATR-based stop is $30&ndash;75 wide. The 0.01 lot minimum then risks
      $30&ndash;75, which is 0.5&ndash;1.25% of the account, so position sizing has
      almost no resolution and some valid setups cannot be taken at all.</p>
    </div>
    <div class="callout">
      <h3>Simulated fills are optimistic by nature</h3>
      <p>Real slippage clusters exactly where these strategies trade &mdash; breakouts
      and news. The modelled $0.15 against stops is a reasonable average, not a bad
      day.</p>
    </div>
  </div>
</section>

<section>
  <div class="section-head prose">
    <p class="eyebrow">If you actually do this</p>
    <h2>What I would tell you</h2>
  </div>
  <div class="prose">
    <ul class="plain">
      <li><b>Do not run four correlated accounts.</b> At {avg_corr} average correlation
      you are paying four fees for close to one bet. Either trade one account, or find
      edges that fail at different times &mdash; which, on gold alone, this search
      could not do.</li>
      <li><b>A3 and A4 are the two worth keeping.</b> They held most of their edge
      into the test window (+0.267 and +0.322 R per trade) and were the only two to
      get funded, while A1 and A2 collapsed to near zero (+0.014 and +0.082).</li>
      <li><b>Expect the challenge to be the easy part.</b> Three of the four passed
      Phase 1 and two got funded in under four months. Staying profitable afterwards
      is where this fell apart, and that is where the money actually is.</li>
      <li><b>Watch the daily limit, not just the total.</b> A3's worst day in the
      funded stage was &minus;4.51% against a 5% hard stop. One more losing trade that
      day and the account was gone.</li>
      <li><b>The reset option is the real cost driver.</b> At $59.99 a reset, a
      strategy that gets funded slowly and dies quickly is worse than one that never
      passes.</li>
    </ul>
  </div>
</section>

<footer>
  <hr>
  <p style="margin-top:22px">Backtest on Dukascopy XAUUSD M1 bid/ask data,
  1 Jan 2025 &ndash; 19 Aug 2026. Simulated results from historical data; they are not
  a prediction and not investment advice.</p>
</footer>
</div>
"""


if __name__ == "__main__":
    html = build()
    out = os.path.join(REPORTS, "report.html")
    with open(out, "w") as fh:
        fh.write(html)
    print(f"wrote {out} ({len(html) / 1024:.0f} KB)")
