# -*- coding: utf-8 -*-
"""每日預測面板（手機優先，單檔 HTML，離線可開）。

設計依 lieflat-charts：
  圖型  G10 Diverging Bar 的資料編碼 —— 零軸居中、正負分向、長度∝|期望值|。
        以手寫 SVG 實作，因為手機面板需離線可開、不依賴 ECharts。
  選型  比較過 F5 Tick Rows（單極、限 ≤8 行）、L2 Dot Cascade（類目名豎排，
        中文長名不適用）、F12 Dumbbell（兩時點對比，非本資料形狀），
        三者都無法誠實編碼雙極資料，故降級至 Glance G10。
  色板  Mono 灰階。50 檔類目遠超 6，依規則不得使用彩色預設。
        方向編碼正負、明度編碼信心度，顏色不承擔第二種含義。
"""
from __future__ import annotations
import datetime as dt, html, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DOCS, DATA
import ranking as ranking_mod

# ── custom 色板：沿用 todo-app 的五色色票（使用者指定）──
#   依 lieflat-charts 第六點五節「使用者明確給出色板時建立 custom」的規則。
#   角色分配：BG 棉紙白／TXT 深藍墨／磚紅＝正期望值（漲）／鴨綠＝負期望值（跌）。
#   依台股慣例紅漲綠跌，而非歐美的綠漲紅跌。
BG, CARD, INK, LINE = "#F8F7F2", "#FFFEFC", "#133E50", "#DCD9CE"
ACCENT, DANGER, HOT, OLIVE = "#2A7574", "#A83F17", "#F4A14F", "#636845"
D_BG, D_CARD, D_INK, D_LINE = "#111214", "#181A1D", "#ECEEF1", "#34383D"
D_ACCENT, D_DANGER = "#5BA7A6", "#E97552"

CSS = f"""
:root{{--bg:{BG};--card:{CARD};--ink:{INK};--mut:#6B7F8A;--faint:#9AA8B0;
--line:{LINE};--pos:{DANGER};--neg:{ACCENT};--hot:{HOT};--r:16px;
--e:cubic-bezier(.2,.8,.25,1)}}
@media(prefers-color-scheme:dark){{:root:not([data-t="light"]){{--bg:{D_BG};--card:{D_CARD};
--ink:{D_INK};--mut:#8A949C;--faint:#5C656C;--line:{D_LINE};
--pos:{D_DANGER};--neg:{D_ACCENT}}}}}
:root[data-t="dark"]{{--bg:{D_BG};--card:{D_CARD};--ink:{D_INK};--mut:#8A949C;
--faint:#5C656C;--line:{D_LINE};--pos:{D_DANGER};--neg:{D_ACCENT}}}
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,-apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC",sans-serif;
font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:560px;margin:0 auto;padding:22px 16px 60px}}
h1{{font-size:20px;font-weight:700;letter-spacing:-.02em;margin:0 0 3px}}
.sub{{font-size:11.5px;color:var(--mut);margin:0 0 16px}}
.read{{background:var(--card);border-radius:var(--r);padding:15px 17px;margin:0 0 14px;
font-size:12.5px;line-height:1.62;box-shadow:0 1px 2px rgba(19,62,80,.07)}}
.read b{{font-weight:700;color:var(--ink)}}
.kpi{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0 0 16px}}
.kpi div{{background:var(--card);border-radius:var(--r);padding:11px 12px;
box-shadow:0 1px 2px rgba(19,62,80,.07)}}
.kpi .k{{font-size:10px;color:var(--mut);letter-spacing:.02em}}
.kpi .v{{font-size:19px;font-weight:800;letter-spacing:-.02em;margin-top:1px;
font-variant-numeric:tabular-nums}}
.kpi div:nth-child(1) .v{{color:var(--pos)}}
.kpi div:nth-child(2) .v{{color:var(--neg)}}
.tabs{{display:flex;gap:6px;margin:0 0 12px}}
.tabs button{{flex:1;padding:9px 0;border:none;border-radius:99px;background:var(--card);
color:var(--mut);font:inherit;font-size:12.5px;font-weight:600;cursor:pointer;
transition:background .14s var(--e),color .14s var(--e)}}
.tabs button[aria-selected="true"]{{background:var(--ink);color:{BG}}}
.list{{background:var(--card);border-radius:var(--r);overflow:hidden;
box-shadow:0 1px 2px rgba(19,62,80,.07)}}
.row{{display:block;width:100%;border:none;background:none;padding:0;
font:inherit;color:inherit;cursor:pointer;text-align:left;
border-bottom:1px solid var(--line)}}
.list .row:last-child{{border-bottom:none}}
.row[aria-expanded="true"]{{background:color-mix(in srgb,var(--ink) 4%,transparent)}}
.top{{display:grid;grid-template-columns:20px 1fr 92px 58px;align-items:center;
gap:7px;padding:9px 13px}}
.rk{{font-size:10px;color:var(--faint);font-variant-numeric:tabular-nums;text-align:right}}
.nm{{font-size:13.5px;font-weight:600;letter-spacing:-.01em;overflow:hidden;
text-overflow:ellipsis;white-space:nowrap}}
.nm span{{font-size:10px;color:var(--mut);font-weight:400;margin-left:4px}}
.ev{{font-size:13px;font-weight:800;text-align:right;font-variant-numeric:tabular-nums;
letter-spacing:-.02em}}
.ev.p{{color:var(--pos)}}.ev.n{{color:var(--neg)}}
.asym{{font-size:9px;color:var(--hot);text-align:right;letter-spacing:.02em;font-weight:600}}
.det{{display:none;padding:2px 13px 14px;font-size:11.5px;color:var(--mut);line-height:1.6}}
.row[aria-expanded="true"] .det{{display:block}}
.det .g{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin:6px 0 9px}}
.det .g div{{background:var(--bg);border-radius:10px;padding:6px 8px}}
.det .g .k{{font-size:9px;color:var(--mut)}}
.det .g .v{{font-size:12.5px;font-weight:700;color:var(--ink);
font-variant-numeric:tabular-nums;margin-top:1px}}
.why{{color:var(--ink);font-size:11.5px}}
.foot{{margin-top:22px;font-size:9.5px;color:var(--faint);letter-spacing:.06em;line-height:1.8}}
.dot{{display:inline-block;width:7px;height:7px;border-radius:2px;vertical-align:-1px;
margin-right:3px}}
.tog{{position:fixed;right:14px;bottom:14px;width:38px;height:38px;border-radius:50%;
border:none;background:var(--card);color:var(--mut);font-size:15px;cursor:pointer;
box-shadow:0 2px 8px rgba(19,62,80,.12)}}
@media(prefers-reduced-motion:no-preference){{
.bar{{transform-origin:var(--o) center;animation:g .5s var(--e) both}}
@keyframes g{{from{{transform:scaleX(0)}}to{{transform:scaleX(1)}}}}}}
"""


def _bar(ev: float, mx: float, conf: str) -> str:
    """G10 編碼：零軸在中線，正值向右、負值向左，長度 ∝ |期望值|。
    明度編碼信心度（high 最黑），不重複編碼數值。"""
    W, H, C = 92, 16, 46
    w = min(abs(ev) / mx * 43, 43) if mx > 0 else 0
    op = {"high": 1.0, "medium": .74, "low": .52}.get(conf, .52)
    col = "var(--pos)" if ev >= 0 else "var(--neg)"
    x = C if ev >= 0 else C - w
    origin = "left" if ev >= 0 else "right"
    return (f'<svg class="b" width="{W}" height="{H}" viewBox="0 0 {W} {H}" aria-hidden="true">'
            f'<line x1="{C}" y1="1.5" x2="{C}" y2="{H-1.5}" stroke="var(--grid)" stroke-width="1"/>'
            f'<rect class="bar" style="--o:{origin}" x="{x}" y="4" width="{max(w,0.8):.1f}" '
            f'height="{H-8}" rx="3" fill="{col}" opacity="{op}"/></svg>')


def _rows(t: pd.DataFrame) -> str:
    if t.empty:
        return '<p class="sub">（尚無判斷）</p>'
    mx = t["exp_ret"].abs().max()
    out = []
    for i, r in t.iterrows():
        ev = float(r["exp_ret"])
        yoy = "—" if pd.isna(r["rev_yoy"]) else f"{r['rev_yoy']*100:+.0f}%"
        fgn = "—" if pd.isna(r["foreign_5"]) else f"{r['foreign_5']:+.2f}"
        asym = str(r["asymmetry"] or "")
        tag = "上檔大" if "正偏" in asym else ("下檔大" if "負偏" in asym else "")
        why = html.escape(str(r["rationale"]).split(": ", 1)[-1])
        out.append(
            f'<button class="row" aria-expanded="false" onclick="t(this)">'
            f'<div class="top"><div class="rk">{i+1}</div>'
            f'<div class="nm">{html.escape(str(r["名稱"]))}<span>{r["code"]}</span></div>'
            f'{_bar(ev, mx, str(r["conviction"]))}'
            f'<div><div class="ev {"p" if ev>=0 else "n"}">{ev*100:+.2f}%</div>'
            f'<div class="asym">{tag}</div></div></div>'
            f'<div class="det"><div class="g">'
            f'<div><div class="k">P(漲)</div><div class="v">{r["prob_up"]*100:.0f}%</div></div>'
            f'<div><div class="k">漲幅</div><div class="v">{r["up_magnitude"]*100:+.1f}%</div></div>'
            f'<div><div class="k">跌幅</div><div class="v">{r["dn_magnitude"]*100:+.1f}%</div></div>'
            f'<div><div class="k">賠率比</div><div class="v">{r["reward_risk"]:.2f}</div></div>'
            f'<div><div class="k">RSI</div><div class="v">{r["rsi_14"]:.0f}</div></div>'
            f'<div><div class="k">營收YoY</div><div class="v">{yoy}</div></div>'
            f'<div><div class="k">外資5日</div><div class="v">{fgn}</div></div>'
            f'<div><div class="k">信心</div><div class="v">{r["conviction"]}</div></div>'
            f'</div><div class="why">{why}</div></div></button>')
    return "".join(out)


def build() -> Path:
    t20, t5 = ranking_mod.table(20), ranking_mod.table(5)
    if t20.empty:
        raise RuntimeError("尚無判斷可呈現")
    as_of = t20["as_of"].iloc[0]
    d = f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
    pos, neg = (t20["exp_ret"] > 0).sum(), (t20["exp_ret"] < 0).sum()
    mean = t20["exp_ret"].mean() * 100

    ctx = ""
    rp = DATA / "reasoning.jsonl"
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                j = json.loads(line)
                if j.get("as_of") == as_of and j.get("market_context"):
                    ctx = j["market_context"]
                    break
    ctx = html.escape(ctx)

    gen = dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=8)))
    doc = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>當日預測 · {d}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div class="wrap">
<h1>當日預測</h1>
<p class="sub">{d} 收盤 · 市值前 50 大 · 期望值 = P(漲)×漲幅 + P(跌)×跌幅</p>

<div class="kpi">
<div><div class="k">正期望值</div><div class="v">{pos}</div></div>
<div><div class="k">負期望值</div><div class="v">{neg}</div></div>
<div><div class="k">全體均值</div><div class="v">{mean:+.2f}%</div></div>
</div>

<div class="read"><b>市場判讀</b><br>{ctx}</div>

<div class="tabs" role="tablist">
<button role="tab" aria-selected="true" onclick="s(this,'h20')">20 個交易日</button>
<button role="tab" aria-selected="false" onclick="s(this,'h5')">5 個交易日</button>
</div>

<div class="list" id="h20">{_rows(t20)}</div>
<div class="list" id="h5" hidden>{_rows(t5)}</div>

<div class="foot">
<span class="dot" style="background:{DANGER}"></span>正期望值（預期上漲）　
<span class="dot" style="background:{ACCENT}"></span>負期望值（預期下跌）<br>
柱長 ∝ 期望值絕對值 · 深淺 = 信心度 · 點任一列展開四面向依據<br>
產生於 {gen:%Y-%m-%d %H:%M} 台北 · 研究與紀律工具，不構成投資建議</div>
</div>
<button class="tog" onclick="k()" aria-label="切換深淺色">◐</button>
<script>
function t(e){{e.setAttribute('aria-expanded',e.getAttribute('aria-expanded')!=='true')}}
function s(b,id){{
 document.querySelectorAll('.tabs button').forEach(x=>x.setAttribute('aria-selected',x===b));
 ['h20','h5'].forEach(x=>document.getElementById(x).hidden=(x!==id));}}
function k(){{const r=document.documentElement;
 const c=r.getAttribute('data-t')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
 const n=c==='dark'?'light':'dark';r.setAttribute('data-t',n);
 try{{localStorage.setItem('t',n)}}catch(e){{}}}}
try{{const v=localStorage.getItem('t');if(v)document.documentElement.setAttribute('data-t',v)}}catch(e){{}}
</script></body></html>"""
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"面板已產生：{p}（{p.stat().st_size/1024:.0f} KB）")
