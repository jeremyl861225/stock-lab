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
.list{background:var(--card);border-radius:var(--r);overflow:hidden;
box-shadow:0 1px 2px rgba(19,62,80,.07)}
.row{display:block;width:100%;border:none;background:none;padding:0;font:inherit;
color:inherit;cursor:pointer;text-align:left;border-bottom:1px solid var(--line)}
.list .row:last-child{border-bottom:none}
.row[aria-expanded="true"]{background:color-mix(in srgb,var(--ink) 4%,transparent)}
.top{display:flex;align-items:center;gap:6px;padding:9px 11px}
.rk{font-size:10px;color:var(--faint);font-variant-numeric:tabular-nums;min-width:16px;
text-align:right}
.nm{font-size:16px;font-weight:800;letter-spacing:-.02em;white-space:nowrap}
.cd{font-size:11.5px;font-weight:600;color:var(--mut);letter-spacing:.02em}
.wt{display:inline-block;width:26px;height:4px;border-radius:2px;background:var(--line);
overflow:hidden;flex-shrink:0}
.wt>span{display:block;height:100%;background:var(--mut);border-radius:2px}
.wp{font-size:9px;color:var(--faint);font-variant-numeric:tabular-nums;min-width:26px}
.bar{margin-left:auto;flex-shrink:0}
.ev{font-size:12.5px;font-weight:800;font-variant-numeric:tabular-nums;
min-width:52px;text-align:right}
.ev.p{color:var(--up)}.ev.n{color:var(--dn)}
.det{display:none;padding:0 11px 11px}
.row[aria-expanded="true"] .det{display:block}
.chips{display:grid;grid-template-columns:repeat(4,1fr);gap:5px}
.c{background:var(--neu);border-radius:9px;padding:5px 6px;text-align:center}
.c b{display:block;font-size:8.5px;font-weight:600;color:var(--mut);letter-spacing:.02em}
.c i{display:block;font-size:13px;font-weight:800;font-style:normal;margin-top:1px;
font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.c.u{background:var(--upbg)}.c.u i{color:var(--up)}
.c.d{background:var(--dnbg)}.c.d i{color:var(--dn)}
.c.h i{color:var(--hot)}
.why{font-size:10.5px;color:var(--mut);line-height:1.55;margin-top:7px}
.hint{font-size:10px;color:var(--faint);margin:0 0 6px;padding-left:2px}
.foot{margin-top:20px;font-size:9.5px;color:var(--faint);letter-spacing:.04em;line-height:1.8}
.dot{display:inline-block;width:7px;height:7px;border-radius:2px;vertical-align:-1px;margin-right:3px}
#pull{position:fixed;top:0;left:0;right:0;height:56px;display:flex;
align-items:center;justify-content:center;gap:6px;font-size:12px;font-weight:600;
color:var(--mut);background:var(--bg);transform:translateY(-56px);z-index:50;
pointer-events:none}
#pull.on{transition:transform .28s var(--e)}
#pull .ico{width:13px;height:13px;border:2px solid var(--line);
border-top-color:var(--ink);border-radius:50%}
#pull.go .ico{animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
@media(prefers-reduced-motion:no-preference){
.bx{transform-origin:var(--o) center;animation:g .5s var(--e) both}
@keyframes g{from{transform:scaleX(0)}to{transform:scaleX(1)}}}
body{overscroll-behavior-y:none}
.tog{position:fixed;right:13px;bottom:13px;width:37px;height:37px;border-radius:50%;
border:none;background:var(--card);color:var(--mut);font-size:15px;cursor:pointer;
box-shadow:0 2px 8px rgba(19,62,80,.12)}
"""


def _bar(ev: float, mx: float, conf: str) -> str:
    """G10 Diverging Bar 的編碼：零軸居中、正負分向、長度 ∝ |期望值|。
    深淺編碼信心度，顏色編碼方向（紅漲綠跌）。"""
    W, H, C = 74, 15, 37
    w = min(abs(ev) / mx * 34, 34) if mx > 0 else 0
    op = {"high": 1.0, "medium": .74, "low": .5}.get(conf, .5)
    col = "var(--up)" if ev >= 0 else "var(--dn)"
    x = C if ev >= 0 else C - w
    o = "left" if ev >= 0 else "right"
    return (f'<svg class="bar" width="{W}" height="{H}" viewBox="0 0 {W} {H}" aria-hidden="true">'
            f'<line x1="{C}" y1="2" x2="{C}" y2="{H-2}" stroke="var(--line)" stroke-width="1"/>'
            f'<rect class="bx" style="--o:{o}" x="{x}" y="4" width="{max(w,.8):.1f}" '
            f'height="{H-8}" rx="3" fill="{col}" opacity="{op}"/></svg>')


def _wt(w: float, mxw: float) -> str:
    """權重條：這檔在該市場市值中的佔比。
    台積電一檔就佔台股 universe 的一半，這件事比任何預測都值得先看到。"""
    if pd.isna(w) or mxw <= 0:
        return ""
    pct = min(w / mxw * 100, 100)
    return (f'<span class="wt" title="市值權重 {w*100:.2f}%">'
            f'<span style="width:{pct:.1f}%"></span></span>')


def _cards(t: pd.DataFrame, cur: str) -> str:
    if t.empty:
        return '<p class="sub">（尚無判斷）</p>'
    mx = t["exp_ret"].abs().max()
    mxw = t["權重"].max() if "權重" in t.columns else 0
    out = []
    for i, r in t.iterrows():
        ev, close = float(r["exp_ret"]), float(r["close"])
        up, dn = float(r["up_magnitude"]), float(r["dn_magnitude"])
        tp, sl = close * (1 + up), close * (1 + dn)
        dec = 2 if close < 100 else (1 if close < 1000 else 0)
        conf = {"high": "高", "medium": "中", "low": "低"}.get(str(r["conviction"]), "低")
        why = html.escape(str(r["rationale"]).split(": ", 1)[-1])
        wt = float(r["權重"]) if "權重" in t.columns and pd.notna(r["權重"]) else float("nan")
        wtxt = "" if pd.isna(wt) else f'<span class="wp">{wt*100:.1f}%</span>'
        out.append(
            f'<button class="row" aria-expanded="false" onclick="t(this)">'
            f'<div class="top"><span class="rk">{i+1}</span>'
            f'<span class="nm">{html.escape(str(r["名稱"]))}</span>'
            f'<span class="cd">{r["code"]}</span>'
            f'{_wt(wt, mxw)}{wtxt}'
            f'{_bar(ev, mx, str(r["conviction"]))}'
            f'<span class="ev {"p" if ev>=0 else "n"}">{ev*100:+.2f}%</span></div>'
            f'<div class="det"><div class="chips">'
            f'<div class="c"><b>P漲</b><i>{r["prob_up"]*100:.0f}%</i></div>'
            f'<div class="c u"><b>漲幅</b><i>{up*100:+.1f}%</i></div>'
            f'<div class="c d"><b>跌幅</b><i>{dn*100:+.1f}%</i></div>'
            f'<div class="c"><b>賠率比</b><i>{r["reward_risk"]:.2f}</i></div>'
            f'<div class="c"><b>收盤價</b><i>{cur}{close:,.{dec}f}</i></div>'
            f'<div class="c u"><b>獲利點</b><i>{cur}{tp:,.{dec}f}</i></div>'
            f'<div class="c d"><b>停損點</b><i>{cur}{sl:,.{dec}f}</i></div>'
            f'<div class="c h"><b>信心</b><i>{conf}</i></div>'
            f'</div><div class="why">{why}</div></div></button>')
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
                 f'<p class="hint">點任一列展開詳細數字　·　細條＝市值權重　·　橫桿＝期望值</p>'
                 f'<div class="list">{_cards(t, cur)}</div></div>')

    gen = dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=8)))
    doc = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#F8F7F2" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#111214" media="(prefers-color-scheme: dark)">
<link rel="manifest" href="manifest.json">
<link rel="apple-touch-icon" href="icons/icon-180.png">
<link rel="icon" type="image/svg+xml" href="icons/icon.svg">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="股市預測">
<title>股市預測 · {d}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<div id="pull"><span class="ico"></span><span id="pulltx">下拉更新</span></div>
<div class="wrap">
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

<div class="foot" data-gen="{gen:%Y-%m-%d %H:%M}">
<span class="dot" style="background:#A83F17"></span>上漲　
<span class="dot" style="background:#2A7574"></span>下跌　（台股慣例紅漲綠跌）<br>
獲利點 = 收盤 ×(1+漲幅)　停損點 = 收盤 ×(1+跌幅)，直接由預測幅度推導<br>
賠率比 = |漲幅 ÷ 跌幅|，偏離 1 代表報酬與風險不對稱<br>
排序依期望值 = P(漲)×漲幅 + P(跌)×跌幅<br>
產生於 {gen:%Y-%m-%d %H:%M} 台北 · 研究與紀律工具，不構成投資建議</div>
</div>
<button class="tog" onclick="k()" aria-label="切換深淺色">◐</button>
<script>
function t(e){{e.setAttribute('aria-expanded',e.getAttribute('aria-expanded')!=='true')}}
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
if('serviceWorker' in navigator)
 navigator.serviceWorker.register('sw.js',{{scope:'./'}}).catch(()=>{{}});

/* 下拉更新。
   PWA 在 standalone 下沒有瀏覽器 UI，原生下拉刷新失效，必須自己做。
   更關鍵的是：Service Worker 走 stale-while-revalidate，直接 reload 只會
   再拿到同一份快取。所以要先用 cache:'reload' 繞過 HTTP 快取抓最新版、
   寫回 SW 快取，再重載 —— 少了這一步，下拉會看起來沒反應。 */
(function(){{
 const bar=document.getElementById('pull'), tx=document.getElementById('pulltx');
 const TH=62; let y0=0, dy=0, on=false, busy=false;
 const set=d=>bar.style.transform='translateY('+(d-56)+'px)';
 const reset=()=>{{bar.classList.add('on');set(0);
   setTimeout(()=>{{bar.classList.remove('on');bar.classList.remove('go');
     tx.textContent='下拉更新';}},280);}};

 addEventListener('touchstart',e=>{{
   if(busy||window.scrollY>0)return;
   y0=e.touches[0].clientY; on=true; dy=0; bar.classList.remove('on');
 }},{{passive:true}});

 addEventListener('touchmove',e=>{{
   if(!on||busy)return;
   dy=e.touches[0].clientY-y0;
   if(dy<=0||window.scrollY>0){{on=false;set(0);return;}}
   set(Math.min(dy*0.45,72));                       // 阻尼，拉不到底
   tx.textContent = dy*0.45>=TH ? '放開更新' : '下拉更新';
 }},{{passive:true}});

 addEventListener('touchend',async()=>{{
   if(!on||busy){{on=false;return;}}
   on=false;
   if(dy*0.45<TH){{reset();return;}}
   busy=true; bar.classList.add('on','go'); set(56); tx.textContent='更新中…';
   const job=(async()=>{{
     try{{
       const r=await fetch('./index.html',{{cache:'reload'}});  // 繞過 HTTP 快取
       if(r&&r.ok&&window.caches){{
         const c=await caches.open('stocklab-v1');
         await c.put('./index.html',r.clone());
         await c.put('./',r.clone());
       }}
     }}catch(e){{}}
   }})();
   // 保險：網路慢或 caches 不可用時，最多等 3 秒仍要重載，否則會卡在「更新中…」
   await Promise.race([job,new Promise(r=>setTimeout(r,3000))]);
   try{{sessionStorage.setItem('refreshed','1')}}catch(e){{}}
   location.reload();
 }},{{passive:true}});

 // 重載後給一次明確回饋 —— 當天資料沒變時，沒有回饋會讓人以為下拉壞了
 try{{
   if(sessionStorage.getItem('refreshed')){{
     sessionStorage.removeItem('refreshed');
     const stamp=document.querySelector('.foot').dataset.gen||'—';
     bar.classList.add('on'); tx.textContent='已更新　資料時間 '+stamp;
     bar.querySelector('.ico').style.display='none';
     set(56);
     setTimeout(()=>{{set(0);setTimeout(()=>{{
       bar.classList.remove('on');bar.querySelector('.ico').style.display='';
       tx.textContent='下拉更新';}},300);}},1800);
   }}
 }}catch(e){{}}
}})();
</script></body></html>"""
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "index.html"
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"面板已產生：{p}（{p.stat().st_size/1024:.0f} KB）")
