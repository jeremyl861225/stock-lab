# -*- coding: utf-8 -*-
"""每日預測面板（手機優先，單檔 HTML，離線可開）。

色板沿用 todo-app 五色（使用者指定）：棉紙白底、深藍墨本文、
磚紅＝上漲、鴨綠＝下跌（台股慣例紅漲綠跌）、琥珀＝信心標記。

每檔顯示八個數字，各自以色塊承載：
  P漲／漲幅／跌幅／年化波動／收盤價／獲利點／停損點／信心
獲利點 = 收盤 ×(1+漲幅)，漲幅是「上漲情境下的平均幅度」，合理可達。
停損點 = 收盤 ×(1+下檔10%分位)，設在正常波動之外，跌破才代表判斷錯了。
年化波動是這兩個價位拉多開的原因，也是全卡唯一與方向判斷無關的數字。
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
.pin{border:none;background:none;padding:0 1px;font-size:12px;line-height:1;
color:var(--line);cursor:pointer;flex-shrink:0;transition:color .14s var(--e)}
.row.pinned .pin{color:var(--hot)}
.row.pinned{background:color-mix(in srgb,var(--hot) 7%,transparent);
box-shadow:inset 3px 0 0 var(--hot)}
.rk{font-size:10px;color:var(--faint);font-variant-numeric:tabular-nums;min-width:16px;
text-align:right}
.nm{font-size:16px;font-weight:800;letter-spacing:-.02em;white-space:nowrap}
.cd{font-size:11.5px;font-weight:600;color:var(--mut);letter-spacing:.02em}
.wt{display:inline-block;width:26px;height:4px;border-radius:2px;background:var(--line);
overflow:hidden;flex-shrink:0}
.wt>span{display:block;height:100%;background:var(--mut);border-radius:2px}
.cps{margin:6px 0 0;display:flex;flex-direction:column;gap:3px}
.cp{display:flex;gap:6px;align-items:baseline;font-size:10.5px;line-height:1.45}
.cp s{text-decoration:none;flex:0 0 11px;font-weight:800}
.cp.ok s{color:var(--up)} .cp.no s{color:var(--dn)} .cp.pd s{color:var(--faint)}
.cp.no em{font-style:normal;color:var(--dn);font-weight:700}
.cp b{font-weight:400;color:var(--mut)} .cp em{font-style:normal;color:var(--faint);
font-variant-numeric:tabular-nums}
.vd{display:inline-block;margin-left:5px;padding:1px 6px;border-radius:99px;
font-size:9px;font-weight:700;letter-spacing:.02em}
.vd.v0{background:var(--up);color:#fff} .vd.v1{background:var(--dn);color:#fff}
.vd.v2{background:var(--line);color:var(--mut)}
.wp{font-size:9px;color:var(--faint);font-variant-numeric:tabular-nums;min-width:26px}
.bar{margin-left:auto;flex-shrink:0}
.ev{font-size:12.5px;font-weight:800;font-variant-numeric:tabular-nums;
min-width:52px;text-align:right}
.ev.p{color:var(--up)}.ev.n{color:var(--dn)}.ev.z{color:var(--mut)}
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
    # 期望值恰為 0 代表「不做方向判斷」，不該畫成微幅看多的紅色。
    if abs(ev) < 1e-9:
        col, op = "var(--mut)", .35
    else:
        col = "var(--up)" if ev > 0 else "var(--dn)"
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


def _score_checkpoints(jdir: Path) -> dict:
    """讀最新的一年期判斷檔，逐檔重評檢查點。評分失敗不能讓整張面板倒 ——
    一年期是附加分頁，5／20 日才是每天要看的東西。"""
    out = {}
    fs = sorted(jdir.glob("*_1y*.json"))
    if not fs:
        return out
    try:
        import checkpoints as CP
        today = pd.Timestamp.today()
        for j in json.loads(fs[-1].read_text(encoding="utf-8")).get("judgments", []):
            try:
                out[str(j["code"])] = CP.score_all(j, today)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return {}
    return out


def _vd(r: dict | None) -> str:
    """論點判定徽章。過半前提被推翻就標失效 —— 該重寫判斷，不是等到期。"""
    if not r:
        return ""
    cls = {"成立": "v0", "失效": "v1", "動搖": "v1"}.get(r["verdict"], "v2")
    return f'<span class="vd {cls}">{r["verdict"]}</span>'


def _cp_html(r: dict | None) -> str:
    """一年期的論點檢查點。沒有檢查點的一年期判斷等於「押方向然後等一年」，
    在這套系統裡沒有價值 —— 所以缺的時候明講，不留白。"""
    if not r:
        return ('<div class="cps"><div class="cp pd"><s>·</s>'
                '<b>此判斷未寫檢查點，一年內無法驗證對錯</b></div></div>')
    mark = {"holding": ("ok", "✓"), "broken": ("no", "✗"), "pending": ("pd", "·")}
    rows = []
    for c in r["checkpoints"]:
        cls, sym = mark[c["status"]]
        act = "待公告" if c["actual"] is None else f'{c["actual"]:+.3f}'
        rows.append(f'<div class="cp {cls}"><s>{sym}</s>'
                    f'<b>{html.escape(str(c["claim"]))}</b>'
                    f'<em>{act}</em></div>')
    return f'<div class="cps">{"".join(rows)}</div>'


def _cards(t: pd.DataFrame, cur: str, hist: dict | None = None, horizon: int = 20,
           cps: dict | None = None) -> str:
    if t.empty:
        return '<p class="sub">（尚無判斷）</p>'
    hist = hist or {}
    mx = t["exp_ret"].abs().max()
    mxw = t["權重"].max() if "權重" in t.columns else 0
    out = []
    for i, r in t.iterrows():
        ev, close = float(r["exp_ret"]), float(r["close"])
        up, dn = float(r["up_magnitude"]), float(r["dn_magnitude"])
        # 獲利點用條件期望漲幅（合理可達的目標）；
        # 停損點用下檔 10% 分位，而非條件期望跌幅 ——
        # 後者正好是下跌情境的中心值，約有一半機率被正常波動掃到，
        # 拿來當停損會被反覆洗出場。
        q10 = float(r["ret_q10"]) if pd.notna(r.get("ret_q10")) else dn * 1.6
        tp, sl = close * (1 + up), close * (1 + q10)
        # 這裡曾放「賠率比」，已移除。它拿條件期望（約五成機率）除以 10% 分位（尾部），
        # 兩者不是同一種量，200 筆全部落在 0.35–0.90、無一 ≥1，等於沒有資訊。
        # 改成同類相比的 |漲幅/跌幅| 也不行：與 P漲 的 R²=85%，只是把旁邊那格換句話說。
        # 改用實證偏態同樣不行：個股偏態前後半期 r=−0.08、符號一致率 55%，不持續。
        # 位置改放年化波動——它持續（前後半期 r=+0.65）、與 P漲 幾乎無關（R²=10%），
        # 而且正是它決定了獲利點與停損點拉多開。
        vol = float(r["vol_20"]) * (252 ** 0.5) if pd.notna(r.get("vol_20")) else float("nan")
        dec = 2 if close < 100 else (1 if close < 1000 else 0)
        conf = {"high": "高", "medium": "中", "low": "低"}.get(str(r["conviction"]), "低")
        # 該檔的歷史準確率。樣本不足時顯示「—」而不是拿 1/1=100% 誤導。
        hc = hist.get(str(r["code"]))
        acc = (f'{hc["weighted"]*100:.0f}%' if hc and hc.get("reliable")
               and hc.get("weighted") is not None else "—")
        accn = f'（{hc["n"]}）' if hc else ""
        why = html.escape(str(r["rationale"]).split(": ", 1)[-1])
        wt = float(r["權重"]) if "權重" in t.columns and pd.notna(r["權重"]) else float("nan")
        wtxt = "" if pd.isna(wt) else f'<span class="wp">{wt*100:.1f}%</span>'
        out.append(
            f'<div class="row" data-code="{r["code"]}" role="button" tabindex="0" '
            f'aria-expanded="false" onclick="t(this)" onkeydown="kd(event,this)">'
            f'<div class="top">'
            f'<button class="pin" aria-label="釘選 {html.escape(str(r["名稱"]))}" '
            f'onclick="pin(event,this)">✦</button>'
            f'<span class="rk">{i+1}</span>'
            f'<span class="nm">{html.escape(str(r["名稱"]))}</span>'
            f'<span class="cd">{r["code"]}</span>'
            f'{_vd((cps or {}).get(str(r["code"])))}'
            f'{_wt(wt, mxw)}{wtxt}'
            f'{_bar(ev, mx, str(r["conviction"]))}'
            f'<span class="ev {"p" if ev>1e-9 else ("n" if ev<-1e-9 else "z")}">{ev*100:+.2f}%</span></div>'
            f'<div class="det"><div class="chips">'
            f'<div class="c"><b>P漲</b><i>{r["prob_up"]*100:.0f}%</i></div>'
            f'<div class="c u"><b>漲幅</b><i>{up*100:+.1f}%</i></div>'
            f'<div class="c d"><b>跌幅</b><i>{dn*100:+.1f}%</i></div>'
            f'<div class="c"><b>年化波動</b><i>{vol*100:.0f}%</i></div>' if vol==vol else '<div class="c"><b>年化波動</b><i>—</i></div>'
            f'<div class="c"><b>收盤價</b><i>{cur}{close:,.{dec}f}</i></div>'
            f'<div class="c u"><b>{"目標價" if horizon >= 250 else "獲利點"}</b>'
            f'<i>{cur}{tp:,.{dec}f}</i></div>'
            # 一年尺度不存在「停損」—— 沒有人抱一年還設日內出場價。
            # 同一個 q10 在這裡的意思是「悲觀情境下的價位」，所以換名字。
            f'<div class="c d"><b>{"保守價" if horizon >= 250 else "停損點"}</b>'
            f'<i>{cur}{sl:,.{dec}f}</i></div>'
            f'<div class="c h"><b>信心·準確</b><i>{conf} {acc}</i></div>'
            f'</div><div class="why">{why}</div>'
            f'{_cp_html((cps or {}).get(str(r["code"]))) if horizon >= 250 else ""}'
            f'</div></div>')
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
        for h in (20, 5, 250):
            t = ranking_mod.table(h, market=mk)
            views[f"{mk}{h}"] = (t, cur)
    if all(v[0].empty for v in views.values()):
        raise RuntimeError("尚無判斷可呈現")

    hist = acc_mod.by_code()
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

    # 一年期的論點檢查點。每次建面板都重評 —— 前提可能昨天還成立、
    # 今天月營收一出就翻掉，這正是這套機制存在的理由。
    cps = _score_checkpoints(jd)

    body = ""
    for key, (t, cur) in views.items():
        mk, hz = key[:2], int(key[2:])
        hint = ('點任一列展開詳細數字　·　點 ✦ 釘選置頂　·　細條＝市值權重　·　橫桿＝期望值'
                if hz < 250 else
                '點任一列展開　·　✓＝前提成立　✗＝已被資料推翻　·＝該期尚未公告')
        body += (f'<div class="view" id="v{key}" hidden>'
                 f'<div class="read"><b>市場判讀</b><br>{html.escape(ctx.get(mk, ""))}</div>'
                 f'<p class="hint">{hint}</p>'
                 f'<div class="list">{_cards(t, cur, hist, hz, cps)}</div></div>')

    gen = dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=8)))
    doc = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#F8F7F2" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#111214" media="(prefers-color-scheme: dark)">
<link rel="manifest" href="manifest.json">
<link rel="apple-touch-icon" sizes="180x180" href="icons/icon-180.png">
<link rel="apple-touch-icon-precomposed" sizes="180x180" href="icons/icon-180.png">
<link rel="icon" type="image/png" sizes="192x192" href="icons/icon-192.png">
<link rel="icon" type="image/png" sizes="512x512" href="icons/icon-512.png">
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
<p class="sub">{d} 收盤 · 台股市值前 50 大 · 美股市值前 50 大＋ETF</p>

{_acc_block(acc_mod.summary())}

<p class="hint" id="pintip" style="margin:0 0 8px"></p>
<div class="tabs" role="tablist">
<button role="tab" aria-selected="true" onclick="m(this,'TW')">台股</button>
<button role="tab" aria-selected="false" onclick="m(this,'US')">美股</button>
</div>
<div class="tabs h" role="tablist">
<button role="tab" aria-selected="true" onclick="z(this,20)">20 日</button>
<button role="tab" aria-selected="false" onclick="z(this,5)">5 日</button>
<button role="tab" aria-selected="false" onclick="z(this,250)">一年</button>
</div>

{body}

<div class="foot" data-gen="{gen:%Y-%m-%d %H:%M}">
<span class="dot" style="background:#A83F17"></span>上漲　
<span class="dot" style="background:#2A7574"></span>下跌　（台股慣例紅漲綠跌）<br>
信心是判斷當下的把握，準確率是該檔過去預測的期望值加權命中率<br>
（樣本未達 8 筆顯示「—」，因為 3 筆對 2 筆不代表 67% 的準確率）<br>
獲利點 = 收盤 ×(1+漲幅)，漲幅為「上漲情境下的平均幅度」<br>
停損點 = 收盤 ×(1+下檔10%分位)，設在正常波動之外，跌破才代表判斷錯了<br>
年化波動 = 近 20 日報酬標準差 ×√252，決定上面兩個價位拉多開<br>
（原「賠率比」已移除：它把五成機率的目標除以一成機率的尾部，200 筆全部 &lt;1，不帶資訊）<br>
排序依期望值 = P(漲)×漲幅 + P(跌)×跌幅<br>
產生於 {gen:%Y-%m-%d %H:%M} 台北 · 研究與紀律工具，不構成投資建議</div>
</div>
<button class="tog" onclick="k()" aria-label="切換深淺色">◐</button>
<script>
function t(e){{e.setAttribute('aria-expanded',e.getAttribute('aria-expanded')!=='true')}}
function kd(e,el){{if(e.key==='Enter'||e.key===' '){{e.preventDefault();t(el);}}}}

/* 釘選：存在瀏覽器本機，只屬於這台裝置的這個瀏覽器，不會外傳。
   釘選的標的置頂，內部仍依期望值排序（用 reverse+prepend 保持相對次序）。 */
let PINS = new Set();
try{{ PINS = new Set(JSON.parse(localStorage.getItem('pins')||'[]')); }}catch(e){{}}
function savePins(){{ try{{ localStorage.setItem('pins', JSON.stringify([...PINS])); }}catch(e){{}} }}
function pin(ev, btn){{
 ev.stopPropagation();                       // 不要連帶展開整列
 const row = btn.closest('.row'), code = row.dataset.code;
 PINS.has(code) ? PINS.delete(code) : PINS.add(code);
 savePins(); applyPins();
}}
function applyPins(){{
 document.querySelectorAll('.list').forEach(list=>{{
   const rows=[...list.querySelectorAll('.row')];
   rows.forEach(r=>r.classList.toggle('pinned', PINS.has(r.dataset.code)));
   // 由後往前 prepend，釘選群組內部維持原本的期望值排序
   rows.filter(r=>PINS.has(r.dataset.code)).reverse().forEach(r=>list.prepend(r));
 }});
 const n=PINS.size, tip=document.getElementById('pintip');
 if(tip) tip.textContent = n ? `已釘選 ${{n}} 檔，置於各分頁最上方` : '';
}}
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
applyPins();
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
