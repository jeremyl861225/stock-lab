# -*- coding: utf-8 -*-
"""每日預測面板（手機優先，單檔 HTML，離線可開）。

色板沿用 todo-app 五色（使用者指定）：棉紙白底、深藍墨本文、
磚紅＝上漲、鴨綠＝下跌（台股慣例紅漲綠跌）、琥珀＝信心標記。

每檔顯示八個數字，各自以色塊承載：
  P漲／漲幅／跌幅／賠率比／收盤價／獲利點／停損點／信心
獲利點 = 收盤 ×(1+漲幅)、停損點 = 收盤 ×(1+跌幅) —— 直接由預測推導，
不是另外設定的目標價，看到的價位就是模型自己說的那個幅度。
"""
from __future__ import annotations
import datetime as dt, html, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DOCS, DATA
import ranking as ranking_mod
import accuracy as acc_mod

CSS = """
:root{--bg:#F8F7F2;--card:#FFFEFC;--ink:#133E50;--mut:#6B7F8A;--faint:#9AA8B0;
--line:#DCD9CE;--up:#A83F17;--dn:#2A7574;--hot:#B9660F;--upbg:#FBE7E0;--dnbg:#DCEBE8;
--neu:#EEEDE7;--r:16px;--e:cubic-bezier(.2,.8,.25,1)}
@media(prefers-color-scheme:dark){:root:not([data-t="light"]){--bg:#111214;--card:#181A1D;
--ink:#ECEEF1;--mut:#8A949C;--faint:#5C656C;--line:#34383D;--up:#E97552;--dn:#5BA7A6;
--hot:#F4A14F;--upbg:#2B1D18;--dnbg:#16292C;--neu:#212429}}
:root[data-t="dark"]{--bg:#111214;--card:#181A1D;--ink:#ECEEF1;--mut:#8A949C;
--faint:#5C656C;--line:#34383D;--up:#E97552;--dn:#5BA7A6;--hot:#F4A14F;
--upbg:#2B1D18;--dnbg:#16292C;--neu:#212429}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;background:var(--bg);color:var(--ink);
font-family:Inter,-apple-system,BlinkMacSystemFont,"Noto Sans TC","PingFang TC",sans-serif;
font-size:15px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:560px;margin:0 auto;padding:20px 14px 60px}
h1{font-size:23px;font-weight:800;letter-spacing:-.03em;margin:0 0 2px}
.sub{font-size:11.5px;color:var(--mut);margin:0 0 14px}
.acc{background:var(--card);border-radius:var(--r);padding:14px 16px;margin:0 0 12px;
box-shadow:0 1px 2px rgba(19,62,80,.07)}
.acc .t{font-size:10.5px;color:var(--mut);letter-spacing:.04em}
.acc .v{font-size:27px;font-weight:800;letter-spacing:-.03em;margin:1px 0 0;
font-variant-numeric:tabular-nums}
.acc .n{font-size:11px;color:var(--mut);margin-top:3px;line-height:1.5}
.tabs{display:flex;gap:5px;margin:0 0 9px}
.tabs button{flex:1;padding:8px 0;border:none;border-radius:99px;background:var(--card);
color:var(--mut);font:inherit;font-size:12.5px;font-weight:700;cursor:pointer;
transition:background .14s var(--e),color .14s var(--e)}
.tabs button[aria-selected="true"]{background:var(--ink);color:var(--bg)}
.tabs.h button{font-size:12px;font-weight:600}
.read{background:var(--card);border-radius:var(--r);padding:13px 15px;margin:0 0 12px;
font-size:11.5px;line-height:1.65;box-shadow:0 1px 2px rgba(19,62,80,.07)}
.read b{font-weight:700}
.card{background:var(--card);border-radius:var(--r);padding:11px 12px 9px;margin:0 0 7px;
box-shadow:0 1px 2px rgba(19,62,80,.07)}
.hd{display:flex;align-items:baseline;gap:7px;margin:0 0 8px}
.rk{font-size:10px;color:var(--faint);font-variant-numeric:tabular-nums;min-width:15px}
.nm{font-size:17px;font-weight:800;letter-spacing:-.02em}
.cd{font-size:12px;font-weight:600;color:var(--mut);letter-spacing:.02em}
.ev{margin-left:auto;font-size:12px;font-weight:800;font-variant-numeric:tabular-nums}
.ev.p{color:var(--up)}.ev.n{color:var(--dn)}
.chips{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}
.c{background:var(--neu);border-radius:9px;padding:5px 6px;text-align:center}
.c b{display:block;font-size:8.5px;font-weight:600;color:var(--mut);letter-spacing:.02em}
.c i{display:block;font-size:13px;font-weight:800;font-style:normal;margin-top:1px;
font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.c.u{background:var(--upbg)}.c.u i{color:var(--up)}
.c.d{background:var(--dnbg)}.c.d i{color:var(--dn)}
.c.h i{color:var(--hot)}
.why{font-size:10.5px;color:var(--mut);line-height:1.55;margin-top:7px}
.foot{margin-top:20px;font-size:9.5px;color:var(--faint);letter-spacing:.04em;line-height:1.8}
.dot{display:inline-block;width:7px;height:7px;border-radius:2px;vertical-align:-1px;margin-right:3px}
.tog{position:fixed;right:13px;bottom:13px;width:37px;height:37px;border-radius:50%;
border:none;background:var(--card);color:var(--mut);font-size:15px;cursor:pointer;
box-shadow:0 2px 8px rgba(19,62,80,.12)}
"""


def _cards(t: pd.DataFrame, cur: str) -> str:
    if t.empty:
        return '<p class="sub">（尚無判斷）</p>'
    out = []
    for i, r in t.iterrows():
        ev, close = float(r["exp_ret"]), float(r["close"])
        up, dn = float(r["up_magnitude"]), float(r["dn_magnitude"])
        tp, sl = close * (1 + up), close * (1 + dn)          # 獲利點／停損點
        dec = 2 if close < 100 else (1 if close < 1000 else 0)
        conf = {"high": "高", "medium": "中", "low": "低"}.get(str(r["conviction"]), "低")
        why = html.escape(str(r["rationale"]).split(": ", 1)[-1])
        out.append(
            f'<div class="card"><div class="hd"><span class="rk">{i+1}</span>'
            f'<span class="nm">{html.escape(str(r["名稱"]))}</span>'
            f'<span class="cd">{r["code"]}</span>'
            f'<span class="ev {"p" if ev>=0 else "n"}">期望 {ev*100:+.2f}%</span></div>'
            f'<div class="chips">'
            f'<div class="c"><b>P漲</b><i>{r["prob_up"]*100:.0f}%</i></div>'
            f'<div class="c u"><b>漲幅</b><i>{up*100:+.1f}%</i></div>'
            f'<div class="c d"><b>跌幅</b><i>{dn*100:+.1f}%</i></div>'
            f'<div class="c"><b>賠率比</b><i>{r["reward_risk"]:.2f}</i></div>'
            f'<div class="c"><b>收盤價</b><i>{cur}{close:,.{dec}f}</i></div>'
            f'<div class="c u"><b>獲利點</b><i>{cur}{tp:,.{dec}f}</i></div>'
            f'<div class="c d"><b>停損點</b><i>{cur}{sl:,.{dec}f}</i></div>'
            f'<div class="c h"><b>信心</b><i>{conf}</i></div>'
            f'</div><div class="why">{why}</div></div>')
    return "".join(out)


def _acc_block(a: dict) -> str:
    if a.get("status") != "ok":
        n = a.get("pending", 0)
        return ('<div class="acc"><div class="t">期望值加權準確率</div>'
                f'<div class="v" style="color:var(--mut)">待結算</div>'
                f'<div class="n">{n} 筆預測已鎖定但尚未到期。'
                '首批 5 日預測於 2026-09-23 結算、20 日於 10-15。<br>'
                '在那之前沒有分數可報 —— 事前鎖死、到期才對答案，是這套系統的重點。</div></div>')
    wh = a.get("weighted_hit")
    parts = []
    for h, d in sorted(a.get("by_horizon", {}).items(), key=lambda x: int(x[0])):
        parts.append(f"{h}日 {d['weighted_hit']*100:.0f}%（{d['n']} 筆）")
    slope = a.get("calib_slope")
    extra = f" · 幅度校準斜率 {slope}" if slope is not None else ""
    return ('<div class="acc"><div class="t">期望值加權準確率</div>'
            f'<div class="v">{wh*100:.1f}%</div>'
            f'<div class="n">未加權命中率 {a["hit_rate"]*100:.1f}% · '
            f'已結算 {a["settled"]} 筆{extra}<br>{" · ".join(parts)}</div></div>')


def build() -> Path:
    views = {}
    for mk, cur in (("TW", "NT$"), ("US", "US$")):
        for h in (20, 5):
            t = ranking_mod.table(h, market=mk)
            views[f"{mk}{h}"] = (t, cur)
    if all(v[0].empty for v in views.values()):
        raise RuntimeError("尚無判斷可呈現")

    any_t = next(v[0] for v in views.values() if not v[0].empty)
    as_of = any_t["as_of"].iloc[0]
    d = f"{as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"

    # 市場判讀直接讀判斷檔，不經 reasoning.jsonl ——
    # 早期的推理記錄沒寫 market 欄位，會讓美股的判讀覆蓋台股。
    ctx = {}
    jd = Path(__file__).resolve().parent.parent / "judgments"
    for f in sorted(jd.glob("*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if j.get("as_of") == as_of and j.get("market_context"):
            ctx[j.get("market", "TW")] = j["market_context"]

    body = ""
    for key, (t, cur) in views.items():
        mk = key[:2]
        body += (f'<div class="view" id="v{key}" hidden>'
                 f'<div class="read"><b>市場判讀</b><br>{html.escape(ctx.get(mk, ""))}</div>'
                 f'{_cards(t, cur)}</div>')

    gen = dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=8)))
    doc = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<title>股市預測 · {d}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div class="wrap">
<h1>股市預測</h1>
<p class="sub">{d} 收盤 · 台股市值前 50 大 · 美股市值前 10 大＋ETF</p>

{_acc_block(acc_mod.summary())}

<div class="tabs" role="tablist">
<button role="tab" aria-selected="true" onclick="m(this,'TW')">台股</button>
<button role="tab" aria-selected="false" onclick="m(this,'US')">美股</button>
</div>
<div class="tabs h" role="tablist">
<button role="tab" aria-selected="true" onclick="z(this,20)">20 個交易日</button>
<button role="tab" aria-selected="false" onclick="z(this,5)">5 個交易日</button>
</div>

{body}

<div class="foot">
<span class="dot" style="background:#A83F17"></span>上漲　
<span class="dot" style="background:#2A7574"></span>下跌　（台股慣例紅漲綠跌）<br>
獲利點 = 收盤 ×(1+漲幅)　停損點 = 收盤 ×(1+跌幅)，直接由預測幅度推導<br>
賠率比 = |漲幅 ÷ 跌幅|，偏離 1 代表報酬與風險不對稱<br>
排序依期望值 = P(漲)×漲幅 + P(跌)×跌幅<br>
產生於 {gen:%Y-%m-%d %H:%M} 台北 · 研究與紀律工具，不構成投資建議</div>
</div>
<button class="tog" onclick="k()" aria-label="切換深淺色">◐</button>
<script>
let MK='TW', HZ=20;
function show(){{
 document.querySelectorAll('.view').forEach(v=>v.hidden=true);
 const el=document.getElementById('v'+MK+HZ); if(el) el.hidden=false;
 window.scrollTo({{top:0,behavior:'instant'}});
}}
function m(b,v){{MK=v;document.querySelectorAll('.tabs:not(.h) button')
 .forEach(x=>x.setAttribute('aria-selected',x===b));show();}}
function z(b,v){{HZ=v;document.querySelectorAll('.tabs.h button')
 .forEach(x=>x.setAttribute('aria-selected',x===b));show();}}
function k(){{const r=document.documentElement;
 const c=r.getAttribute('data-t')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
 const n=c==='dark'?'light':'dark';r.setAttribute('data-t',n);
 try{{localStorage.setItem('t',n)}}catch(e){{}}}}
try{{const v=localStorage.getItem('t');if(v)document.documentElement.setAttribute('data-t',v)}}catch(e){{}}
show();
</script></body></html>"""
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"面板已產生：{p}（{p.stat().st_size/1024:.0f} KB）")
