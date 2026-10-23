"""Build the ScoreBridge graphics from model_results.json (stdlib only).

Writes build/showcase.html and build/logo.html (render them to PNG with render.js) and the vector
logo ../scorebridge_logo.svg.
"""
import json
from pathlib import Path

from brand import mark, square_svg, tile_svg

HERE = Path(__file__).resolve().parent
BUILD = HERE / 'build'
BUILD.mkdir(exist_ok=True)

results = json.load(open(HERE / 'model_results.json'))

# Dark-mode ordinal tier ramp (validated: monotone L, single hue, every step >= 3:1 on the #161922 panel)
TIER = {'Poor': '#B83A5B', 'Standard': '#E0505A', 'Good': '#F8906F'}

row = results['final']  # final customer-level model of Notebook 4 / README
auc = {k: row['auc'][k] for k in ('Good', 'Poor', 'Standard')}
proba = results['sample']['proba']
feats = results['sample']['features']

# ---- ROC chart (plot area W x H; legend sits in the empty region under the chance line) ----
W, H = 176, 140


def roc_path(points):
    return 'M' + ' L'.join(f'{x * W:.1f},{(1 - y) * H:.1f}' for x, y in points)


roc_lines = []
for cls in ('Standard', 'Poor', 'Good'):  # hero series (Good) drawn last, on top
    d = roc_path(row['roc'][cls])
    if cls == 'Good':
        roc_lines.append(f'<path d="{d} L{W},{H} L0,{H} Z" fill="{TIER[cls]}" opacity="0.10"/>')
    roc_lines.append(f'<path d="{d}" fill="none" stroke="{TIER[cls]}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')

legend_rows = []
for i, c in enumerate(('Good', 'Poor', 'Standard')):
    y = H - 46 + i * 16
    legend_rows.append(f'<line x1="{W - 98}" y1="{y - 3.5}" x2="{W - 86}" y2="{y - 3.5}" stroke="{TIER[c]}" '
                       f'stroke-width="2" stroke-linecap="round"/>'
                       f'<text x="{W - 80}" y="{y}" class="lgt">{c}</text>'
                       f'<text x="{W - 4}" y="{y}" class="lgv" text-anchor="end">{auc[c]:.2f}</text>')

roc_svg = f'''<svg class="roc" viewBox="-28 -4 {W + 32} {H + 22}" width="{W + 32}" height="{H + 22}">
  <line x1="0" y1="{H / 2}" x2="{W}" y2="{H / 2}" class="grid"/>
  <line x1="{W / 2}" y1="0" x2="{W / 2}" y2="{H}" class="grid"/>
  <line x1="0" y1="{H}" x2="{W}" y2="0" class="chance"/>
  {''.join(roc_lines)}
  <line x1="0" y1="0" x2="0" y2="{H}" class="axis"/><line x1="0" y1="{H}" x2="{W}" y2="{H}" class="axis"/>
  <text x="-6" y="5" class="tick" text-anchor="end">1</text>
  <text x="-6" y="{H + 2}" class="tick" text-anchor="end">0</text>
  <text x="-6" y="{H / 2 + 3}" class="tick" text-anchor="end">TPR</text>
  <text x="{W}" y="{H + 14}" class="tick" text-anchor="end">1</text>
  <text x="{W / 2}" y="{H + 14}" class="tick" text-anchor="middle">FPR</text>
  <rect x="{W - 106}" y="{H - 60}" width="108" height="56" rx="6" class="lgbox"/>
  {''.join(legend_rows)}
</svg>'''

# ---- Class-probability meters (Good first: it's the predicted tier) ----
meters = ''.join(
    f'''<div><div class="mrow"><span><i style="background:{TIER[c]}"></i>{c}</span>
    <b>{proba[c] * 100:.0f}%</b></div><div class="track" style="--c:{TIER[c]}">
    <div class="fill" style="width:{proba[c] * 100:.1f}%"></div></div></div>'''
    for c in ('Good', 'Standard', 'Poor'))

# ---- Global feature importance (nominal -> one colour for every bar) ----
NICE = {'Outstanding_Debt': 'Outstanding debt', 'Credit_Mix': 'Credit mix',
        'Payment_of_Min_Amount': 'Min. payments', 'Delay_from_due_date': 'Payment delay',
        'Annual_Income': 'Annual income', 'Total_EMI_per_month': 'Monthly EMI',
        'Num_of_Delayed_Payment': 'Delayed payments', 'Age': 'Age', 'Num_Bank_Accounts': 'Bank accounts',
        'Num_of_Loan': 'Number of loans', 'Changed_Credit_Limit': 'Credit-limit change'}
top = results['importances'][:5]
vmax = top[0][1]
drivers = ''.join(
    f'''<div class="drv"><span>{NICE[n]}</span><div class="dbar"><div style="width:{v / vmax * 100:.1f}%">
    </div></div><b>{v * 100:.0f}%</b></div>''' for n, v in top)

pred = max(proba, key=proba.get)
facts = [('Credit mix', feats['Credit_Mix']), ('Debt', f"{feats['Outstanding_Debt']:,.0f}"),
         ('Days late', f"{feats['Delay_from_due_date']:.0f}"),
         ('Min. pay only', feats['Payment_of_Min_Amount'])]
facts_html = ''.join(f'<div class="fact"><span>{k}</span><b>{v}</b></div>' for k, v in facts)

gauge = (f'<svg class="gauge" viewBox="12 25 76 47">'
         f'{mark(colors=(TIER["Poor"], TIER["Standard"], TIER["Good"]), needle="#F4F2F5", w=8, gap=22, needle_w=3.6, needle_len=20, pivot=5)}</svg>')

acc = row['test_acc'] * 100
base = row['baseline'] * 100

showcase = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>ScoreBridge showcase</title>
<style>
:root {{
  --page:#0D0F15; --window:#12141B; --panel:#161922; --line:rgba(255,255,255,.075);
  --grid:#252935; --t1:#F4F2F5; --t2:#A9AEBC; --t3:#7D8294;
  --coral:#F05941; --crimson:#BE3144; --wine:#872341;
}}
* {{ box-sizing:border-box; margin:0; }}
html,body {{ width:1200px; height:628px; overflow:hidden; }}
body {{
  background:
    radial-gradient(560px 420px at 83% 30%, rgba(190,49,68,.30), transparent 70%),
    radial-gradient(420px 300px at 100% 100%, rgba(240,89,65,.14), transparent 70%),
    radial-gradient(500px 400px at 0% 0%, rgba(135,35,65,.18), transparent 70%),
    var(--page);
  color:var(--t1); font-family:Inter, sans-serif; position:relative;
  -webkit-font-smoothing:antialiased;
}}
body::before {{ /* faint dot grid, fading out from the top right */
  content:""; position:absolute; inset:0; pointer-events:none;
  background-image:radial-gradient(rgba(255,255,255,.07) 1px, transparent 1.2px);
  background-size:22px 22px;
  -webkit-mask-image:radial-gradient(700px 520px at 78% 35%, #000 0%, transparent 75%);
}}
.left {{ position:absolute; left:56px; top:48px; width:520px; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.brand .name {{ font:600 21px "Inter Display", Inter; letter-spacing:-.02em; }}
.brand .sep {{ width:1px; height:18px; background:var(--line); margin:0 4px; }}
.brand .kind {{ font:500 11px "JetBrains Mono"; letter-spacing:.08em; text-transform:uppercase; color:var(--t3); }}
.eyebrow {{ margin-top:46px; display:inline-flex; align-items:center; gap:8px; padding:6px 12px 6px 10px;
  border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.03);
  font:500 11px "JetBrains Mono"; letter-spacing:.08em; text-transform:uppercase; color:var(--t2); }}
.eyebrow i {{ width:6px; height:6px; border-radius:50%; background:var(--coral);
  box-shadow:0 0 0 3px rgba(240,89,65,.18); }}
h1 {{ margin-top:18px; font:600 64px/1.0 "Inter Display", Inter; letter-spacing:-.045em; }}
h1 em {{ font-style:normal; background:linear-gradient(90deg,#F8906F 0%,#F05941 45%,#BE3144 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; }}
.lead {{ margin-top:18px; max-width:470px; font-size:16px; line-height:1.55; color:var(--t2); }}
.lead strong {{ color:var(--t1); font-weight:500; }}
.stack {{ margin-top:22px; display:flex; flex-wrap:wrap; gap:7px; }}
.stack span {{ font:500 11px "JetBrains Mono"; color:var(--t2); padding:5px 9px; border-radius:7px;
  border:1px solid var(--line); background:rgba(255,255,255,.025); }}
.kpis {{ position:absolute; left:56px; top:478px; width:556px; display:grid;
  grid-template-columns:repeat(4,1fr); }}
.kpi {{ padding:0 14px; border-left:1px solid var(--line); }}
.kpi:first-child {{ padding-left:0; border-left:0; }}
.kpi b {{ display:block; font:600 30px Inter; letter-spacing:-.03em; }}
.kpi span {{ display:block; margin-top:4px; font-size:11.5px; line-height:1.35; color:var(--t3); }}
.foot {{ position:absolute; left:56px; right:48px; bottom:22px; display:flex; justify-content:space-between;
  font:500 10.5px "JetBrains Mono"; letter-spacing:.04em; color:var(--t3); }}
.foot b {{ color:var(--t2); font-weight:500; }}

/* ---------- product window ---------- */
.win {{ position:absolute; left:616px; top:44px; width:536px; height:534px; border-radius:18px;
  background:var(--window); border:1px solid rgba(255,255,255,.09);
  box-shadow:0 40px 90px rgba(0,0,0,.55), 0 0 0 1px rgba(0,0,0,.4), inset 0 1px 0 rgba(255,255,255,.05); }}
.bar {{ height:42px; display:flex; align-items:center; padding:0 16px; gap:7px;
  border-bottom:1px solid var(--line); }}
.bar .dot {{ width:9px; height:9px; border-radius:50%; background:#2B2F3B; }}
.bar .title {{ margin-left:12px; font:500 11px "JetBrains Mono"; letter-spacing:.03em; color:var(--t2); }}
.bar .pill {{ margin-left:auto; font:500 10px "JetBrains Mono"; letter-spacing:.05em; color:var(--t2);
  padding:4px 8px; border-radius:6px; border:1px solid var(--line); }}
.grid {{ display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr; gap:12px;
  padding:14px; height:calc(100% - 42px); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }}
.panel.a {{ grid-row:1 / span 2; display:flex; flex-direction:column; justify-content:space-between; }}
.ph {{ display:flex; justify-content:space-between; align-items:center;
  font:500 10px "JetBrains Mono"; letter-spacing:.08em; text-transform:uppercase; color:var(--t3); }}
.who {{ margin-top:7px; font-size:12.5px; color:var(--t2); }}
.who b {{ color:var(--t1); font-weight:500; }}
.gauge {{ display:block; width:168px; height:104px; margin:0 auto; }}
.pred {{ text-align:center; margin-top:6px; }}
.pred span {{ font:500 10px "JetBrains Mono"; letter-spacing:.08em; text-transform:uppercase; color:var(--t3); }}
.pred b {{ display:block; font:600 28px Inter; letter-spacing:-.03em; margin-top:2px; }}
.pred em {{ font-style:normal; font-size:12.5px; color:var(--t2); }}
.meters {{ display:grid; gap:9px; }}
.mrow {{ display:flex; justify-content:space-between; font-size:12px; color:var(--t2); }}
.mrow span {{ display:flex; align-items:center; gap:7px; }}
.mrow i {{ width:7px; height:7px; border-radius:2px; }}
.mrow b {{ color:var(--t1); font-weight:500; font-variant-numeric:tabular-nums; }}
.track {{ margin-top:5px; height:6px; border-radius:1px 3px 3px 1px;
  background:color-mix(in srgb, var(--c) 18%, var(--panel)); }}
.fill {{ height:100%; border-radius:1px 3px 3px 1px; background:var(--c); }}
.facts {{ padding-top:12px; border-top:1px solid var(--line);
  display:grid; grid-template-columns:1fr 1fr; gap:7px 18px; }}
.fact {{ display:flex; justify-content:space-between; align-items:baseline; font-size:11px; }}
.fact span {{ color:var(--t3); }}
.fact b {{ font-size:11.5px; font-weight:500; }}
.rocwrap {{ margin-top:12px; }}
.roc .grid {{ stroke:var(--grid); stroke-width:1; }}
.roc .chance {{ stroke:#3A3F4D; stroke-width:1; }}
.roc .axis {{ stroke:#3A3F4D; stroke-width:1; }}
.roc .tick {{ font:500 8.5px "JetBrains Mono"; fill:var(--t3); }}
.roc .lgbox {{ fill:var(--panel); stroke:var(--line); }}
.roc .lgt {{ font:500 9.5px Inter; fill:var(--t1); }}
.roc .lgv {{ font:500 9.5px Inter; fill:var(--t2); font-variant-numeric:tabular-nums; }}
.panel.c {{ display:flex; flex-direction:column; }}
.drivers {{ margin-top:16px; display:grid; gap:14px; }}
.drv {{ display:grid; grid-template-columns:116px 1fr 30px; align-items:center; gap:8px; font-size:11.5px; color:var(--t2); }}
.drv b {{ text-align:right; color:var(--t1); font-weight:500; font-variant-numeric:tabular-nums; }}
.dbar {{ height:6px; }}
.dbar div {{ height:100%; border-radius:0 3px 3px 0; background:#C9CDD8; }}
.note {{ margin-top:auto; font-size:10px; color:var(--t3); }}
</style></head>
<body>
  <div class="left">
    <div class="brand">{tile_svg(34, uid='hdr')}<span class="name">ScoreBridge</span>
      <span class="sep"></span><span class="kind">ML case study</span></div>
    <div class="eyebrow"><i></i>Multiclass credit-score classification</div>
    <h1>Credit risk,<br><em>classified.</em></h1>
    <p class="lead">An end-to-end ML pipeline that turns <strong>100K messy customer records</strong> into
      <strong>Poor · Standard · Good</strong> credit-score predictions to promote financial inclusion.</p>
    <div class="stack"><span>Python</span><span>pandas</span><span>scikit-learn</span>
      <span>{row['model']}</span><span>GroupKFold CV</span></div>
  </div>
  <div class="kpis">
    <div class="kpi"><b>100K</b><span>customer records<br>cleaned &amp; modelled</span></div>
    <div class="kpi"><b>6</b><span>model families<br>benchmarked</span></div>
    <div class="kpi"><b>{auc['Good']:.2f}</b><span>ROC-AUC<br>(Good tier)</span></div>
    <div class="kpi"><b>{acc:.0f}%</b><span>accuracy on new<br>customers vs {base:.0f}%</span></div>
  </div>
  <div class="foot"><span><b>Vassilis Tsoubris</b> · Machine Learning</span>
    <span>github.com/vtsou359/Credit_Score_Classification_Project</span></div>

  <div class="win">
    <div class="bar"><span class="dot"></span><span class="dot"></span><span class="dot"></span>
      <span class="title">ScoreBridge · Applicant review</span>
      <span class="pill">MODEL · {row['model'].upper()}</span></div>
    <div class="grid">
      <section class="panel a">
        <div><div class="ph"><span>Sample applicant</span><span>Unseen</span></div>
        <div class="who">Age <b>{feats['Age']:.0f}</b> · Income <b>{feats['Annual_Income'] / 1000:.1f}K</b> / yr</div></div>
        <div>{gauge}
        <div class="pred"><span>Predicted tier</span><b>{pred}</b>
          <em>{proba[pred] * 100:.0f}% probability</em></div></div>
        <div class="meters">{meters}</div>
        <div class="facts">{facts_html}</div>
      </section>
      <section class="panel">
        <div class="ph"><span>ROC · one-vs-rest</span><span>Unseen</span></div>
        <div class="rocwrap">{roc_svg}</div>
      </section>
      <section class="panel c">
        <div class="ph"><span>Top drivers</span><span>Importance</span></div>
        <div class="drivers">{drivers}</div>
        <div class="note">Impurity-based importance · {row['n_features']} model inputs</div>
      </section>
    </div>
  </div>
</body></html>'''

logo = ('<!doctype html><html><head><meta charset="utf-8"><title>ScoreBridge logo</title>'
        '<style>*{margin:0} html,body{width:600px;height:600px;overflow:hidden} svg{display:block}</style>'
        f'</head><body>{square_svg(600)}</body></html>')

(BUILD / 'showcase.html').write_text(showcase)
(BUILD / 'logo.html').write_text(logo)
(HERE.parent / 'scorebridge_logo.svg').write_text(tile_svg(512) + '\n')
print('wrote build/showcase.html, build/logo.html, ../scorebridge_logo.svg')
