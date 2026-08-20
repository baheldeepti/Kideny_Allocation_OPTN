#!/usr/bin/env python3
"""Render PAPER.md into a self-contained, theme-aware HTML artifact with the
figures embedded as base64 data URIs (Artifact CSP blocks external assets)."""
import base64, re
from pathlib import Path
import markdown

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
OUT = BASE / "paper.html"

def datauri(name):
    b = (FIG / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(b).decode()

md_text = (BASE / "PAPER.md").read_text()
body_md = md_text[md_text.find("## Abstract"):]
html_body = markdown.markdown(
    body_md, extensions=["tables", "fenced_code", "sane_lists"])

# add pass/fail coloring to check/cross marks in tables
html_body = html_body.replace("✓", '<span class="ok">✓</span>')
html_body = html_body.replace("✗", '<span class="no">✗</span>')

# Convert the Markdown-inlined figures (src="figures/NAME.png") into styled,
# base64-embedded <figure> blocks so the page is self-contained (CSP-safe).
# The GitHub-rendered PAPER.md uses the same figures/ paths directly.
def _embed_figure(m):
    alt, name = m.group(1), m.group(2)
    tag, _, cap = alt.partition(". ")
    if not cap:                       # no "Fig N." prefix
        tag, cap = "", alt
    return (f'<figure><img alt="{cap}" src="{datauri(name)}">'
            f'<figcaption><span class="figtag">{tag}</span> {cap}'
            f'</figcaption></figure>')

html_body = re.sub(
    r'<p><img alt="([^"]*)" src="figures/([^"]+)"\s*/?></p>',
    _embed_figure, html_body)

KPIS = [
    ("494,862", "kidney-alone registrations", "2015+ cohort"),
    ("14.1%", "adverse-outcome rate", "died / too-sick"),
    ("0.726", "ROC-AUC", "calibrated XGBoost"),
    ("0.212 → 0.111", "Brier score", "after isotonic calibration"),
    ("0.629", "C-index", "time-to-transplant"),
    ("2.51×", "top-decile lift", "risk concentration"),
]
kpi_html = "".join(
    f'<div class="kpi"><div class="kpi-val">{v}</div>'
    f'<div class="kpi-lab">{l}</div><div class="kpi-sub">{s}</div></div>'
    for v, l, s in KPIS)

CSS = """
<style>
:root{
  --paper:#F6F7F9; --card:#FFFFFF; --ink:#1B2A33; --muted:#5C6B73;
  --accent:#0E7C86; --accent-ink:#0B5960; --rule:#E2E6EA; --rule-soft:#EDF0F2;
  --ok:#2F855A; --no:#C0392B; --band:#F0F3F4; --shadow:rgba(20,40,50,.07);
}
@media (prefers-color-scheme:dark){:root{
  --paper:#0E1519; --card:#141E24; --ink:#DCE3E7; --muted:#8A9AA3;
  --accent:#3BB0BA; --accent-ink:#6FCBD3; --rule:#243138; --rule-soft:#1B262C;
  --ok:#57C08A; --no:#E27A6E; --band:#16222A; --shadow:rgba(0,0,0,.35);
}}
:root[data-theme="light"]{
  --paper:#F6F7F9; --card:#FFFFFF; --ink:#1B2A33; --muted:#5C6B73;
  --accent:#0E7C86; --accent-ink:#0B5960; --rule:#E2E6EA; --rule-soft:#EDF0F2;
  --ok:#2F855A; --no:#C0392B; --band:#F0F3F4; --shadow:rgba(20,40,50,.07);
}
:root[data-theme="dark"]{
  --paper:#0E1519; --card:#141E24; --ink:#DCE3E7; --muted:#8A9AA3;
  --accent:#3BB0BA; --accent-ink:#6FCBD3; --rule:#243138; --rule-soft:#1B262C;
  --ok:#57C08A; --no:#E27A6E; --band:#16222A; --shadow:rgba(0,0,0,.35);
}
*{box-sizing:border-box}
.wrap{
  --serif:"Iowan Old Style","Charter","Palatino Linotype",Georgia,"Times New Roman",serif;
  --sans:ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:var(--paper); color:var(--ink); font-family:var(--serif);
  font-size:18px; line-height:1.7; -webkit-font-smoothing:antialiased;
  padding:0 20px 96px; min-height:100vh;
}
.col{max-width:760px; margin:0 auto}
/* hero */
.hero{max-width:900px; margin:0 auto; padding:64px 0 28px; border-bottom:1px solid var(--rule)}
.eyebrow{font-family:var(--sans); font-size:12.5px; font-weight:650;
  letter-spacing:.14em; text-transform:uppercase; color:var(--accent)}
h1.title{font-size:clamp(28px,4.4vw,44px); line-height:1.12; font-weight:700;
  letter-spacing:-.01em; text-wrap:balance; margin:.5rem 0 .4rem}
.sub{color:var(--muted); font-size:19px; max-width:60ch; margin:.2rem 0 0}
.meta{font-family:var(--sans); font-size:13px; color:var(--muted);
  margin-top:20px; display:flex; flex-wrap:wrap; gap:6px 20px}
.meta b{color:var(--ink); font-weight:600}
/* kpis */
.kpis{max-width:900px; margin:26px auto 0; display:grid;
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:1px;
  background:var(--rule); border:1px solid var(--rule); border-radius:12px; overflow:hidden}
.kpi{background:var(--card); padding:16px 18px}
.kpi-val{font-family:var(--sans); font-weight:700; font-size:22px;
  font-variant-numeric:tabular-nums; letter-spacing:-.01em; color:var(--accent-ink)}
.kpi-lab{font-family:var(--sans); font-size:13px; font-weight:600; margin-top:3px}
.kpi-sub{font-family:var(--sans); font-size:12px; color:var(--muted); margin-top:1px}
/* body */
.body{padding-top:8px}
.body h2{font-size:25px; font-weight:700; letter-spacing:-.01em; text-wrap:balance;
  margin:56px 0 4px; padding-top:20px; border-top:1px solid var(--rule-soft)}
.body h2:first-of-type{border-top:0; margin-top:28px}
.body h3{font-family:var(--sans); font-size:15px; font-weight:680; text-transform:uppercase;
  letter-spacing:.05em; color:var(--muted); margin:34px 0 2px}
.body p{margin:14px 0}
.body strong{font-weight:680}
.body a{color:var(--accent); text-decoration:none; border-bottom:1px solid var(--rule)}
.body a:hover{border-color:var(--accent)}
.body ul,.body ol{padding-left:1.3em; margin:14px 0}
.body li{margin:7px 0}
.body code{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace; font-size:.82em;
  background:var(--band); padding:.12em .4em; border-radius:5px; color:var(--accent-ink)}
.body hr{border:0; border-top:1px solid var(--rule); margin:40px 0}
/* abstract */
.body > h2:first-of-type + p{font-size:19px}
/* tables */
.tablewrap{overflow-x:auto; margin:18px 0; border:1px solid var(--rule); border-radius:10px}
.body table{width:100%; border-collapse:collapse; font-family:var(--sans);
  font-size:14px; font-variant-numeric:tabular-nums}
.body thead th{background:var(--band); text-align:left; font-weight:650; color:var(--ink);
  padding:10px 14px; border-bottom:1px solid var(--rule); white-space:nowrap}
.body tbody td{padding:9px 14px; border-bottom:1px solid var(--rule-soft)}
.body tbody tr:last-child td{border-bottom:0}
.body tbody tr:hover{background:var(--band)}
.ok{color:var(--ok); font-weight:700}
.no{color:var(--no); font-weight:700}
/* figures */
.figures{margin:52px 0 8px; padding-top:20px; border-top:1px solid var(--rule-soft)}
.fig-grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; margin-top:14px}
figure{margin:24px auto; max-width:620px; background:var(--card);
  border:1px solid var(--rule); border-radius:12px;
  overflow:hidden; box-shadow:0 1px 3px var(--shadow)}
figure img{display:block; width:100%; height:auto; background:#fff}
figcaption{font-family:var(--sans); font-size:12.5px; color:var(--muted);
  line-height:1.5; padding:11px 14px 13px}
.figtag{font-weight:700; color:var(--accent-ink)}
/* footer */
.foot{max-width:900px; margin:56px auto 0; padding-top:20px; border-top:1px solid var(--rule);
  font-family:var(--sans); font-size:12.5px; color:var(--muted)}
@media (max-width:560px){.wrap{font-size:16.5px} .hero{padding-top:40px}}
</style>
"""

HTML = f"""{CSS}
<div class="wrap">
  <header class="hero">
    <div class="eyebrow">OPTN kidney registry · predictive modeling</div>
    <h1 class="title">Predicting Waitlist Risk and Time-to-Transplant for Kidney Candidates</h1>
    <p class="sub">A reproducible, fairness-aware analysis of national OPTN registry data — who is at risk, how long they wait, and whether the model is fair and explainable.</p>
    <div class="meta">
      <span><b>Cohort</b> 2015+ kidney-alone waitlist</span>
      <span><b>Data</b> OPTN STAR KIDPAN · 202606</span>
      <span><b>Tasks</b> classification + survival</span>
      <span><b>Draft</b> conference · 2026-08-19</span>
    </div>
  </header>
  <div class="kpis">{kpi_html}</div>
  <main class="col body">
    {html_body}
  </main>
  <footer class="col foot">
    Aggregates and figures only — no row-level records, consistent with the OPTN Data Use Agreement.
    Reproducible from <code>build_analytic_extract.py</code> and <code>run_analysis.py</code> on the 202606 STAR release.
  </footer>
</div>
"""

# wrap markdown tables in a horizontal-scroll container
HTML = HTML.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")

OUT.write_text(HTML)
print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
