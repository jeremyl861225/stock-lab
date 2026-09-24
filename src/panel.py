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
import datetime as dt, html, json, re, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DOCS, DATA
import ranking as ranking_mod
from models.price_1y import price as price_1y
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
/* 市場判讀：原生 <details>，預設收合。它一天 1,800 字，在手機上是兩個螢幕高，
   展開才看得到清單；收合時只留一行預覽，開合偏好記在本機（見 rd()）。 */
.read{background:var(--card);border-radius:var(--r);padding:0;margin:0 0 12px;
font-size:11.5px;line-height:1.65;box-shadow:0 1px 2px rgba(19,62,80,.07)}
.read b{font-weight:700}
.read summary{list-style:none;cursor:pointer;display:flex;align-items:center;gap:8px;
padding:12px 15px;user-select:none;-webkit-user-select:none}
.read summary::-webkit-details-marker{display:none}
.read summary b{flex:0 0 auto;font-size:12px}
.rd-pv{flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
color:var(--mut)}
.rd-ch{flex:0 0 auto;margin-left:auto;width:14px;text-align:center;color:var(--faint);
font-size:11px;transition:transform .18s var(--e)}
.read[open] .rd-ch{transform:rotate(180deg)}
.read[open] .rd-pv{display:none}
.rd-body{padding:0 15px 13px}
.rd-body p{margin:0 0 8px}.rd-body p:last-child{margin:0}
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
/* 名稱可以被壓縮並截斷；期望值不行 —— 它是整列最重要的數字。
   原本 .nm 只有 nowrap 沒有 overflow，長名（美股 "Goldman Sachs Group, Inc. (The)"）
   會把整列撐寬，把期望值推出螢幕右緣。台股名 2–4 字所以看不出來。 */
.nm{font-size:16px;font-weight:800;letter-spacing:-.02em;white-space:nowrap;
overflow:hidden;text-overflow:ellipsis;min-width:0;flex:0 1 auto}
.cd{font-size:11.5px;font-weight:600;color:var(--mut);letter-spacing:.02em;flex:0 0 auto}
/* 美股：代號當主體（不可壓縮），公司名當副標（可截斷） */
.nm.tk{flex:0 0 auto;letter-spacing:0}
.cd.sub{flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;
white-space:nowrap;font-weight:500}
.wt{display:inline-block;width:26px;height:4px;border-radius:2px;background:var(--line);
overflow:hidden;flex-shrink:0}
.wt>span{display:block;height:100%;background:var(--mut);border-radius:2px}
/* K 線。四個時間尺度，展開後才抓資料、才渲染 ——
   103 檔 × 4 個尺度若在載入時全部畫出來，是 412 張 SVG。 */
.kw{margin:8px 0 2px}
.ks{display:flex;gap:4px;margin:0 0 6px}
.ks button{flex:1;padding:4px 0;border:none;border-radius:7px;background:var(--bg);
color:var(--mut);font:inherit;font-size:10.5px;font-weight:600;cursor:pointer;
transition:background .14s var(--e),color .14s var(--e)}
.ks button.on{background:var(--ink);color:var(--bg)}
.kbox{position:relative;width:100%;height:132px;background:var(--bg);border-radius:9px;
overflow:hidden}
.kbox svg{display:block;width:100%;height:100%}
.kg{stroke:var(--line);stroke-width:.5}
.kup{fill:var(--up);stroke:var(--up)}
.kdn{fill:var(--dn);stroke:var(--dn)}
.khit{fill:transparent;cursor:pointer}
.kbar.sel .kb{stroke-width:1.6}
.klab{fill:var(--faint);font-size:7px;font-weight:600}
.ktip{margin:5px 0 0;min-height:15px;font-size:10.5px;color:var(--mut);
font-variant-numeric:tabular-nums;line-height:1.4}
.ktip b{color:var(--ink);font-weight:700}
.ktip .u{color:var(--up);font-weight:700}.ktip .d{color:var(--dn);font-weight:700}
.kmsg{display:flex;align-items:center;justify-content:center;height:100%;
font-size:10.5px;color:var(--faint)}
.cps{margin:6px 0 0;display:flex;flex-direction:column;gap:3px}
.cp{display:flex;gap:6px;align-items:baseline;font-size:10.5px;line-height:1.45}
.cp s{text-decoration:none;flex:0 0 11px;font-weight:800}
.cp.ok s{color:var(--up)} .cp.no s{color:var(--dn)} .cp.pd s{color:var(--faint)}
.cp.no em{font-style:normal;color:var(--dn);font-weight:700}
.cp b{font-weight:400;color:var(--mut)} .cp em{font-style:normal;color:var(--faint);
font-variant-numeric:tabular-nums}
/* 論點判定不用徽章 —— 徽章佔 40px 橫向空間，正好是中文名字剩下的量，
   會把「南亞科」擠成「南…」。改成左緣色條：零橫向成本，
   而且一整欄掃下去比分散的徽章更容易看出哪些論點壞了。 */
.row[data-vd]{border-left:3px solid transparent}
/* 刻意不用 --up／--dn。這個 app 是紅漲綠跌，紅色條會被讀成「上漲」
   而不是「論點成立」—— 兩種語意撞在同一個顏色上。
   改用與價格方向無關的軸：實心深色＝論點站著，琥珀＝動搖，淡灰＝已倒。 */
.row[data-vd="成立"]{border-left-color:var(--ink)}
.row[data-vd="動搖"]{border-left-color:var(--hot)}
.row[data-vd="失效"]{border-left-color:var(--faint)}
.row[data-vd="失效"] .nm{color:var(--mut)}
.vdt{font-size:10px;font-weight:700;letter-spacing:.04em;margin:0 0 4px}
.vsrc{display:block;margin-top:2px;font-size:9.5px;font-weight:400;color:var(--faint);
letter-spacing:0}
.vdt.v0{color:var(--ink)} .vdt.v1{color:var(--hot)} .vdt.v2{color:var(--mut)}
.wp{font-size:9px;color:var(--faint);font-variant-numeric:tabular-nums;min-width:26px;flex:0 0 auto}
.bar{margin-left:auto;flex-shrink:0}
.ev{font-size:12.5px;font-weight:800;font-variant-numeric:tabular-nums;
min-width:52px;text-align:right;flex:0 0 auto;margin-left:auto}
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
    # 橫桿從 74px 縮到 58px：390px 的手機上，固定元素原本吃掉 290px，
    # 只剩 78px 給名稱。橫桿是相對比較用的，短一點不影響判讀。
    W, H, C = 58, 15, 29
    w = min(abs(ev) / mx * 26, 26) if mx > 0 else 0
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
    # 必須逐市場各取最新一份。字典序下 20260918_us_1y.json 排在
    # 20260918_1y.json 之後，只取 [-1] 會讓台股的檢查點整組消失 ——
    # 而且是靜默的：面板照樣渲染，只是所有台股都變成「待驗」。
    latest: dict[str, tuple] = {}
    for f in jdir.glob("*_1y*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "judgments" not in d:
            continue
        mk = d.get("market", "TW")
        if mk not in latest or str(d.get("as_of", "")) > str(latest[mk][0]):
            latest[mk] = (str(d.get("as_of", "")), d)
    if not latest:
        return out
    try:
        import checkpoints as CP
        today = pd.Timestamp.today()
        for _, doc in latest.values():
          for j in doc.get("judgments", []):
            try:
                r = CP.score_all(j, today, j.get("market") or doc.get("market"))
                # 滾動出處：一年期每天都會重新定價，但論點本身多數日子沒動。
                # 不標出「這個論點是哪天下的」，讀者會以為今天有新判斷。
                r["thesis_as_of"] = j.get("thesis_as_of")
                r["repriced_only"] = j.get("repriced_only")
                r["news_flag"] = j.get("news_flag")
                out[str(j["code"])] = r
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        return {}
    return out


# 美股名稱帶法律後綴，在手機上佔掉一半列寬而不帶任何資訊。
# 只改顯示，不動 briefing 裡的原始名稱。
_SUFFIX = [" (The)", ", Inc.", " Inc.", ", Ltd.", " Ltd.", ", L.P.", " Corporation",
           " Corp.", " Incorporated", " Holdings", " Company", " & Co.", " Co.",
           " Group", " plc", " N.V.", " S.A.", " Class A", " Class B",
           # 資料來源有些名稱被截斷過，句點已不見（"Philip Morris International Inc"）
           " Inc", " Corp", " Ltd", " New", " Trust", " Series 1"]


def _short(name: str) -> str:
    """去掉不帶資訊的法律後綴。反覆剝除，因為常見組合是疊加的
    （"Goldman Sachs Group, Inc. (The)" 要剝三層）。
    剝到剩不到 3 個字就停 —— 寧可長一點也不要剝成認不出來。"""
    n = name.strip()
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIX:
            if n.endswith(suf) and len(n) - len(suf) >= 2:   # 3M、GE、HP 都是合法的兩字名
                n = n[: -len(suf)].rstrip(" ,&")
                # "Eli Lilly and Company" 剝掉 Company 會留下懸空的 and
                if n.endswith(" and"):
                    n = n[:-4].rstrip(" ,&")
                changed = True
    return n or name


def _ident(code: str, name: str) -> str:
    """名稱與代號誰當主體，依市場而定。

    台股名稱 2–4 字（台積電），名稱當主體、代號當副標最好認。
    美股名稱長達 31 字（Taiwan Semiconductor Manufactur），
    在 390px 的手機上只剩 78px 給它 —— 會截成「Micron…」「Broa…」，
    反而認不出來。而美股的識別主體本來就是代號（MU／AVGO／NVDA），
    所以對調：代號當主體，公司名當可截斷的副標。
    判準用代號是否全為數字（台股代號一律數字）。
    """
    tw = code.isdigit()
    short = html.escape(_short(name))
    if tw:
        return (f'<span class="nm">{short}</span>'
                f'<span class="cd">{html.escape(code)}</span>')
    return (f'<span class="nm tk">{html.escape(code)}</span>'
            f'<span class="cd sub">{short}</span>')


def _vd(r: dict | None) -> str:
    """列的左緣色條屬性。過半前提被推翻就標失效 —— 該重寫判斷，不是等到期。"""
    return f' data-vd="{r["verdict"]}"' if r else ""


def _vdt(r: dict | None) -> str:
    """展開後的判定文字，連同成立／推翻的條數。"""
    if not r:
        return ""
    cls = {"成立": "v0", "動搖": "v1", "失效": "v2"}.get(r["verdict"], "v2")
    n = f'{r["holding"]} 條成立'
    if r["broken"]:
        n += f"、{r['broken']} 條已被推翻"
    if r.get("unmet"):
        n += f"、{r['unmet']} 條轉機尚未發生"
    if r["pending"]:
        n += f"、{r['pending']} 條尚未公告"
    src = ""
    t = r.get("thesis_as_of")
    if t:
        d = f"{t[:4]}-{t[4:6]}-{t[6:]}"
        src = (f'<span class="vsrc">論點 {d} 起未變，今日僅重新定價</span>'
               if r.get("repriced_only") else
               f'<span class="vsrc">論點 {d} 更新</span>')
    return f'<div class="vdt {cls}">論點{r["verdict"]}　{n}{src}</div>'


def _cp_html(r: dict | None) -> str:
    """一年期的論點檢查點。沒有檢查點的一年期判斷等於「押方向然後等一年」，
    在這套系統裡沒有價值 —— 所以缺的時候明講，不留白。"""
    if not r:
        return ('<div class="cps"><div class="cp pd"><s>·</s>'
                '<b>此判斷未寫檢查點，一年內無法驗證對錯</b></div></div>')
    mark = {"holding": ("ok", "✓"), "broken": ("no", "✗"), "pending": ("pd", "·"),
            "未達成": ("pd", "○")}
    rows = []
    for c in r["checkpoints"]:
        cls, sym = mark[c["status"]]
        act = "待公告" if c["actual"] is None else f'{c["actual"]:+.3f}'
        rows.append(f'<div class="cp {cls}"><s>{sym}</s>'
                    f'<b>{html.escape(str(c["claim"]))}</b>'
                    f'<em>{act}</em></div>')
    return f'<div class="cps">{"".join(rows)}</div>'


def _price_chips(horizon: int, r, close: float, up: float, q10: float,
                 cur: str, dec: int) -> str:
    """價格格。5／20 日與一年期顯示的東西刻意不同。

    5／20 日：獲利點（條件期望漲幅）與停損點（下檔 10% 分位）。
    這兩個是可執行的價位 —— 期間夠短，波動度不會把它們撐到荒謬。

    一年期：**中位價**與**五成區間**（q25–q75）。
    2026-09-19 換掉原本的「目標價／保守價」，理由是實測：
      · 目標價與 σ 的 Spearman 是 +0.972，與 P漲 只有 +0.045。
        把 38 檔的 P漲 全部換成同一個值，目標價的橫斷面差異只掉 1%
        —— 那個數字有 97% 是波動度的讀數，不是判斷。
      · 而它**不是太寬**：過去一年南電實際高/低 7.9 倍、南亞科 10.1 倍、
        華邦電 9.1 倍，而模型的 q75/q25 只有約 3 倍。把它縮窄會讓它
        同時變成沒用又錯的數字。
    所以改的是「顯示什麼」不是「分布」：
      中位價 = 第 50 百分位，median > 0 ⟺ P漲 > 50%，這一格才是判斷；
      五成區間 = 一年後有一半機率落在這裡，是一句說得清楚的話，
      而「目標價／保守價」會被讀成可執行的價位。
    """
    if horizon < 250:
        tp, sl = close * (1 + up), close * (1 + q10)
        return (f'<div class="c u"><b>獲利點</b><i>{cur}{tp:,.{dec}f}</i></div>'
                f'<div class="c d"><b>停損點</b><i>{cur}{sl:,.{dec}f}</i></div>')
    p_up = float(r["prob_up"])
    sig = r.get("sigma_annual")
    if pd.isna(sig) or not sig:
        # 舊帳本沒有 sigma_annual 的列：退回只顯示中位價，不硬湊一個區間。
        med = close * (1 + float(r["exp_ret"]))
        return (f'<div class="c u"><b>中位價</b><i>{cur}{med:,.{dec}f}</i></div>'
                f'<div class="c"><b>五成區間</b><i>—</i></div>')
    q = price_1y(p_up, float(sig))
    med = close * (1 + q["median"])
    lo, hi = close * (1 + q["q25"]), close * (1 + q["q75"])
    return (f'<div class="c u"><b>中位價</b><i>{cur}{med:,.{dec}f}</i></div>'
            f'<div class="c" title="一年後有一半機率落在這個範圍；寬度由波動度決定，不是判斷">'
            f'<b>五成區間</b><i>{lo:,.{dec}f}–{hi:,.{dec}f}</i></div>')


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
        # 價位（獲利點／停損點／中位價）以判斷日的收盤為基準；收盤價那格顯示最新值。
        # 兩者不同日時（某市場當天沒做新判斷），在收盤價格子標明日期，
        # 否則卡片會把 9/22 的收盤配 9/21 的幅度，21 檔價位最多偏 5%（FLOW-05）。
        cj = r.get("close_judged")
        close_j = float(cj) if pd.notna(cj) else close
        b_as = r.get("briefing_as_of")
        stale = isinstance(b_as, str) and b_as != str(r["as_of"])
        up, dn = float(r["up_magnitude"]), float(r["dn_magnitude"])
        # 獲利點用條件期望漲幅（合理可達的目標）；
        # 停損點用下檔 10% 分位，而非條件期望跌幅 ——
        # 後者正好是下跌情境的中心值，約有一半機率被正常波動掃到，
        # 拿來當停損會被反覆洗出場。
        q10 = float(r["ret_q10"]) if pd.notna(r.get("ret_q10")) else dn * 1.6
        # 這裡曾放「賠率比」，已移除。它拿條件期望（約五成機率）除以 10% 分位（尾部），
        # 兩者不是同一種量，200 筆全部落在 0.35–0.90、無一 ≥1，等於沒有資訊。
        # 改成同類相比的 |漲幅/跌幅| 也不行：與 P漲 的 R²=85%，只是把旁邊那格換句話說。
        # 改用實證偏態同樣不行：個股偏態前後半期 r=−0.08、符號一致率 55%，不持續。
        # 位置改放年化波動——它持續（前後半期 r=+0.65）、與 P漲 幾乎無關（R²=10%），
        # 而且正是它決定了獲利點與停損點拉多開。
        # 波動窗口必須跟幅度用的同一個，否則同一張卡上會出現互相矛盾的數字：
        # 2330 的 vol_20 年化 18.0%、vol_60 年化 36.2%（近 20 天剛好平靜），
        # 而一年期的幅度是用 vol_60 算的 —— 卡片若顯示 18% 而幅度是 ±33%，
        # 讀者無從對帳。
        vcol = "vol_60" if horizon >= 250 else "vol_20"
        vraw = r.get(vcol)
        if pd.isna(vraw):
            vraw = r.get("vol_20")
        vol = float(vraw) * (252 ** 0.5) if pd.notna(vraw) else float("nan")
        # 一年期的幅度自 2026-09-19 起用 vol_60 與長期波動的混合 σ 算
        # （models/price_1y.py），卡片就要顯示那個 σ，否則又是兩個對不上的數字。
        if horizon >= 250 and pd.notna(r.get("sigma_annual")):
            vol = float(r["sigma_annual"])
        # 一定要先算成字串再放進 f-string 鏈。把 `A if c else B` 直接寫在
        # 隱式字串串接裡，Python 會把條件套用到「整條鏈」而不是那一格 ——
        # 條件成立時整張卡在這裡截斷、div 不閉合，版面全垮。已中招一次。
        voltxt = f"{vol * 100:.0f}%" if vol == vol else "—"
        dec = 2 if close < 100 else (1 if close < 1000 else 0)
        conf = {"high": "高", "medium": "中", "low": "低"}.get(str(r["conviction"]), "低")
        # 該檔的歷史準確率。樣本不足時顯示「—」而不是拿 1/1=100% 誤導。
        hc = hist.get(str(r["code"]))
        acc = (f'{hc["weighted"]*100:.0f}%' if hc and hc.get("reliable")
               and hc.get("weighted") is not None else "—")
        accn = f'（{hc["n"]}）' if hc else ""
        cp = (cps or {}).get(str(r["code"]))
        why = html.escape(str(r["rationale"]).split(": ", 1)[-1])
        wt = float(r["權重"]) if "權重" in t.columns and pd.notna(r["權重"]) else float("nan")
        wtxt = "" if pd.isna(wt) else f'<span class="wp">{wt*100:.1f}%</span>'
        out.append(
            f'<div class="row" data-code="{r["code"]}"'
            # 色條只掛在一年期 —— 論點狀態是一年期判斷的屬性，
            # 拿它去染 5／20 日的列會讓兩種期別的訊息混在一起。
            f'{_vd(cp) if horizon >= 250 else ""} role="button" tabindex="0" '
            f'aria-expanded="false" onclick="t(this)" onkeydown="kd(event,this)">'
            f'<div class="top">'
            f'<button class="pin" aria-label="釘選 {html.escape(_short(str(r["名稱"])))}" '
            f'onclick="pin(event,this)">✦</button>'
            f'<span class="rk">{i+1}</span>'
            f'{_ident(str(r["code"]), str(r["名稱"]))}'
            f'{_wt(wt, mxw)}{wtxt}'
            f'{_bar(ev, mx, str(r["conviction"]))}'
            f'<span class="ev {"p" if ev>1e-9 else ("n" if ev<-1e-9 else "z")}">{ev*100:+.2f}%</span></div>'
            f'<div class="det"><div class="chips">'
            f'<div class="c"><b>P漲</b><i>{r["prob_up"]*100:.0f}%</i></div>'
            f'<div class="c u"><b>漲幅</b><i>{up*100:+.1f}%</i></div>'
            f'<div class="c d"><b>跌幅</b><i>{dn*100:+.1f}%</i></div>'
            f'<div class="c"><b>年化波動</b><i>{voltxt}</i></div>'
            f'<div class="c"><b>收盤價{("（" + b_as[4:6].lstrip("0") + "/" + b_as[6:].lstrip("0") + "）") if stale else ""}</b>'
            f'<i>{cur}{close:,.{dec}f}</i></div>'
            f'{_price_chips(horizon, r, close_j, up, q10, cur, dec)}'
            f'<div class="c h"><b>信心·準確</b><i>{conf} {acc}</i></div>'
            f'</div><div class="why">{why}</div>'
            f'<div class="kw" data-kc="{r["code"]}">'
            f'<div class="ks" role="tablist">'
            f'<button class="on" data-s="d20" onclick="ks(event,this)">20 天</button>'
            f'<button data-s="m3" onclick="ks(event,this)">3 個月</button>'
            f'<button data-s="y1" onclick="ks(event,this)">1 年</button>'
            f'<button data-s="y5" onclick="ks(event,this)">5 年</button>'
            f'</div><div class="kbox"><div class="kmsg">載入中…</div></div>'
            f'<div class="ktip"></div></div>'
            f'{(_vdt(cp) + _cp_html(cp)) if horizon >= 250 else ""}'
            f'</div></div>')
    return "".join(out)


def _market_context(jd: Path, view_as_of: dict[str, str]) -> dict[tuple[str, str], str]:
    """回傳 {(市場, 期別桶): 判讀}，桶只有兩種：'d' = 5／20 日、'y' = 一年期。

    為什麼要分桶：同一個 as_of 有三份判斷檔（h20、h5、_1y）各帶自己的
    market_context。原本用 {市場: 文字} 收，glob 排序後最後一份蓋掉前面的 ——
    一年期分頁顯示的其實是 5／20 日的判讀，一年期自己寫的那份從來沒上過面板。
    5 日與 20 日共用同一份（DAILY.md：兩者由同一批 p20 推導，判讀本來就是同一段）。
    """
    out: dict[tuple[str, str], str] = {}
    for f in sorted(jd.glob("*.json")):
        try:
            j = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        mk = j.get("market", "TW")
        if not j.get("market_context") or j.get("as_of") != view_as_of.get(mk):
            continue
        bucket = "y" if int(j.get("horizon", 20)) >= 250 else "d"
        # 5／20 日兩份文字相同；若哪天不同，以 20 日那份為準（h5 檔不覆蓋）。
        if bucket == "d" and f.name.endswith("h5.json") and (mk, "d") in out:
            continue
        out[(mk, bucket)] = j["market_context"]
    return out


def _ctx_html(text: str) -> str:
    """判斷檔的 market_context 是輕量 markdown：**粗體** 與空行分段。
    先 escape 再轉，粗體只認成對的雙星號，不會讓內容注入標籤。"""
    t = html.escape(text or "").replace("\r\n", "\n")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t, flags=re.S)
    paras = [p.strip() for p in re.split(r"\n\s*\n", t) if p.strip()]
    return "".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paras)


def _read_html(text: str) -> str:
    """市場判讀區塊：<details> 收合，summary 留一行預覽（去掉 markdown 記號）。"""
    if not text:
        return ('<details class="read"><summary><b>市場判讀</b>'
                '<span class="rd-pv">今日尚無判讀</span></summary></details>')
    first = re.sub(r"\*\*", "", text.strip().split("\n")[0])
    pv = html.escape(first[:90])
    return (f'<details class="read" ontoggle="rd(this)"><summary><b>市場判讀</b>'
            f'<span class="rd-pv">{pv}</span><span class="rd-ch">▾</span></summary>'
            f'<div class="rd-body">{_ctx_html(text)}</div></details>')


def _acc_block(a: dict) -> str:
    if a.get("status") != "ok":
        n = a.get("pending", 0)
        return ('<div class="acc"><div class="t">期望值加權準確率</div>'
                f'<div class="v" style="color:var(--mut)">待結算</div>'
                f'<div class="n">{n} 筆預測已鎖定但尚未到期。'
                '首批 5 日預測於 2026-09-23 結算、20 日於 10-15。<br>'
                '在那之前沒有分數可報 —— 事前鎖死、到期才對答案，是這套系統的重點。</div></div>')
    wh = a.get("weighted_hit")
    parts, extra = [], ""
    for h, d in sorted(a.get("by_horizon", {}).items(), key=lambda x: int(x[0])):
        parts.append(f"{h}日 {d['weighted_hit']*100:.0f}%（{d['n_scored']} 筆）")
        mz = d.get("mz") or {}
        # 選股層只印 rank IC（去均值、對平移不變）；MZ 斜率要有效天數夠才印，
        # 單日的斜率是雜訊（首批 −0.746 的 95% CI 是 [−6.6, +4.6]）。
        if mz.get("rank_ic") is not None:
            extra += f" · {h}日選股 IC {mz['rank_ic']:+.2f}"
            if mz.get("slope") is not None:
                extra += f"（幅度斜率 {mz['slope']:+.2f}）"
        extra += (f" · {h}日市場層 預測 {d['mean_pred']*100:+.2f}% / 實際 "
                  f"{d['mean_actual']*100:+.2f}%")
    ab = f"、{a['abstain']} 筆中性不計" if a.get("abstain") else ""
    hit = a.get("hit_rate")
    hit_txt = f"{hit*100:.1f}%" if hit is not None else "—"
    return ('<div class="acc"><div class="t">期望值加權準確率</div>'
            f'<div class="v">{wh*100:.1f}%</div>'
            f'<div class="n">未加權命中率 {hit_txt} · '
            f'已結算 {a["settled"]} 筆{ab}{extra}<br>{" · ".join(parts)}</div></div>')


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
    jd = Path(__file__).resolve().parent.parent / "judgments"
    # 每個市場用自己的基準日。台股 13:30 收盤、美股隔天清晨才收，兩邊的 as_of
    # 常差一天；原本用台股的 as_of 去找美股判斷檔，美股那格會整天空白。
    view_as_of: dict[str, str] = {}
    for key, (t, _) in views.items():
        if not t.empty:
            view_as_of.setdefault(key[:2], t["as_of"].iloc[0])
    ctx = _market_context(jd, view_as_of)

    # 一年期的論點檢查點。每次建面板都重評 —— 前提可能昨天還成立、
    # 今天月營收一出就翻掉，這正是這套機制存在的理由。
    cps = _score_checkpoints(jd)

    body = ""
    for key, (t, cur) in views.items():
        mk, hz = key[:2], int(key[2:])
        hint = ('點任一列展開詳細數字　·　點 ✦ 釘選置頂　·　細條＝市值權重　·　橫桿＝期望值'
                if hz < 250 else
                '點任一列展開　·　左緣色條＝論點狀態（深＝成立／琥珀＝動搖／淡＝失效）　·　✓前提成立　✗已被推翻')
        body += (f'<div class="view" id="v{key}" hidden>'
                 f'{_read_html(ctx.get((mk, "y" if hz >= 250 else "d"), ""))}'
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
獲利點 = 收盤 ×(1+漲幅)，漲幅為「上漲情境下的平均幅度」（5／20 日）<br>
停損點 = 收盤 ×(1+下檔10%分位)，設在正常波動之外，跌破才代表判斷錯了（5／20 日）<br>
年化波動 = 近 20 日報酬標準差 ×√252，決定上面兩個價位拉多開<br>
（原「賠率比」已移除：它把五成機率的目標除以一成機率的尾部，200 筆全部 &lt;1，不帶資訊）<br>
排序依期望值 = P(漲)×漲幅 + P(跌)×跌幅<br>
<b>一年期分頁不顯示獲利點與停損點</b>，改為中位價與五成區間。原因是實測：
目標價與 σ 的相關是 +0.97、與 P漲 只有 +0.05，把所有標的的 P漲 換成同一個值，
目標價的差異只掉 1% —— 那是波動度的讀數不是判斷。
而它並不是太寬：過去一年南電實際高/低 7.9 倍、南亞科 10.1 倍，模型只有約 3 倍，
所以縮窄會讓它同時變成沒用又錯的數字。改的是顯示什麼，不是分布<br>
<b>中位價</b> = 第 50 百分位，中位價高於收盤 ⟺ P漲 &gt; 50%，這一格才是判斷<br>
<b>五成區間</b> = 一年後有一半機率落在這裡，寬度由波動度決定<br>
一年期的期望值也是中位數（不是平均數 —— 平均數含 exp(σ²/2)，
會讓排序退化成純波動度排序）<br>
產生於 {gen:%Y-%m-%d %H:%M} 台北 · 研究與紀律工具，不構成投資建議</div>
</div>
<button class="tog" onclick="k()" aria-label="切換深淺色">◐</button>
<script>
function t(e){{
 const open = e.getAttribute('aria-expanded')!=='true';
 e.setAttribute('aria-expanded', open);
 if(open) kinit(e);          /* 展開才畫圖 —— 見 kload 的說明 */
}}
function kd(e,el){{if(e.key==='Enter'||e.key===' '){{e.preventDefault();t(el);}}}}
/* 市場判讀的開合偏好。三個市場×期別的區塊同步：在台股開了，切到美股也是開的。
   存本機，只屬於這台裝置。沒存過＝收合（它一天 1,800 字，預設展開會把清單推到兩個螢幕外）。 */
let RDSYNC=false;
function rd(d){{
 if(RDSYNC) return;
 RDSYNC=true;
 document.querySelectorAll('details.read').forEach(x=>{{ if(x!==d) x.open=d.open; }});
 RDSYNC=false;
 try{{localStorage.setItem('rd', d.open?'1':'0')}}catch(e){{}}
}}
try{{ if(localStorage.getItem('rd')==='1'){{ RDSYNC=true;
 document.querySelectorAll('details.read').forEach(x=>x.open=true); RDSYNC=false; }} }}catch(e){{}}

/* ── K 線 ────────────────────────────────────────────────────────
   資料放在同目錄的 charts.json，第一次展開任一列時才抓（約 160 KB 壓縮後），
   抓完存在記憶體。不內嵌進本頁的理由：103 檔 × 4 個尺度 = 412 組序列，
   內嵌會讓每個人不論看不看圖都先背下整包。
   不用圖表庫的理由：外部腳本受 CSP 限制，且為了四種 K 線載入一整個
   函式庫，在手機上不划算 —— SVG 直接畫就夠。                       */
let KD=null, KP=null;
function kload(){{
 if(KD) return Promise.resolve(KD);
 if(!KP) KP = fetch('charts.json').then(r=>r.ok?r.json():Promise.reject(r.status))
   .then(j=>{{KD=j; return j;}});
 return KP;
}}
function kinit(row){{
 const w = row.querySelector('.kw');
 if(!w || w.dataset.done) return;
 w.dataset.done = '1';
 kload().then(()=>kdraw(w)).catch(()=>{{
   const b=w.querySelector('.kbox');
   if(b) b.innerHTML='<div class="kmsg">K 線資料載入失敗</div>';
 }});
}}
function ks(ev, btn){{
 ev.stopPropagation();          /* 不要連帶收合整列 */
 const w = btn.closest('.kw');
 w.querySelectorAll('.ks button').forEach(b=>b.classList.toggle('on', b===btn));
 w.querySelector('.ktip').innerHTML='';
 kdraw(w);
}}
function kfmt(v){{
 return v>=1000 ? v.toLocaleString(undefined,{{maximumFractionDigits:0}})
      : v>=100  ? v.toFixed(1) : v.toFixed(2);
}}
function kdraw(w){{
 const box = w.querySelector('.kbox');
 const code = w.dataset.kc;
 const span = (w.querySelector('.ks button.on')||{{dataset:{{}}}}).dataset.s || 'd20';
 const s = KD && KD.series[code] && KD.series[code][span];
 if(!s){{ box.innerHTML='<div class="kmsg">此區間無資料</div>'; return; }}
 const n=s.close.length, W=300, H=132, PL=2, PR=26, PT=8, PB=12;
 const lo=Math.min(...s.low), hi=Math.max(...s.high), rng=(hi-lo)||1;
 const iw=(W-PL-PR)/n, bw=Math.max(1.2, iw*0.62);
 const y=v=>PT+(hi-v)/rng*(H-PT-PB);
 let g='';
 /* 水平參考線：最高、最低、以及中間值。只畫三條，多了會蓋過 K 線本身 */
 [hi, (hi+lo)/2, lo].forEach(v=>{{
   g+=`<line class="kg" x1="${{PL}}" y1="${{y(v).toFixed(1)}}" x2="${{W-PR}}" y2="${{y(v).toFixed(1)}}"/>`
     +`<text class="klab" x="${{W-PR+2}}" y="${{(y(v)+2.5).toFixed(1)}}">${{kfmt(v)}}</text>`;
 }});
 for(let i=0;i<n;i++){{
   const o=s.open[i],h=s.high[i],l=s.low[i],c=s.close[i];
   const cx=PL+iw*(i+0.5), cls=c>=o?'kup':'kdn';
   const yo=y(o), yc=y(c), top=Math.min(yo,yc), bh=Math.max(0.8,Math.abs(yc-yo));
   const tap = span==='d20' ? ` data-i="${{i}}" onclick="ktap(event,this)"` : '';
   g+=`<g class="kbar"${{tap}}>`
     +`<line class="kb ${{cls}}" x1="${{cx.toFixed(1)}}" y1="${{y(h).toFixed(1)}}" x2="${{cx.toFixed(1)}}" y2="${{y(l).toFixed(1)}}" stroke-width="1"/>`
     +`<rect class="kb ${{cls}}" x="${{(cx-bw/2).toFixed(1)}}" y="${{top.toFixed(1)}}" width="${{bw.toFixed(1)}}" height="${{bh.toFixed(1)}}"/>`
     + (span==='d20'
        ? `<rect class="khit" x="${{(cx-iw/2).toFixed(1)}}" y="0" width="${{iw.toFixed(1)}}" height="${{H}}"/>`
        : '')
     +`</g>`;
 }}
 box.innerHTML=`<svg viewBox="0 0 ${{W}} ${{H}}" preserveAspectRatio="none" role="img" `
   +`aria-label="${{code}} ${{span}} K 線">${{g}}</svg>`;
 const tip=w.querySelector('.ktip');
 tip.innerHTML = span==='d20'
   ? '點任一根 K 棒看當日數字'
   : ({{m3:'日 K · 近 3 個月', y1:'週 K · 近 1 年', y5:'月 K · 近 5 年'}})[span];
}}
function ktap(ev, g){{
 ev.stopPropagation();          /* 不要連帶收合整列 */
 const w = g.closest('.kw'), i = +g.dataset.i;
 w.querySelectorAll('.kbar').forEach(x=>x.classList.toggle('sel', x===g));
 const s = KD.series[w.dataset.kc].d20, p = s.p[i];
 const cls = p==null ? '' : (p>0?'u':(p<0?'d':''));
 const sign = p==null ? '—' : (p>0?'+':'')+p.toFixed(2)+'%';
 w.querySelector('.ktip').innerHTML =
   `<b>${{s.t[i]}}</b>　開 ${{kfmt(s.open[i])}}　高 ${{kfmt(s.high[i])}}　`
  +`低 ${{kfmt(s.low[i])}}　收 <b>${{kfmt(s.close[i])}}</b>　`
  +`<span class="${{cls}}">${{sign}}</span>`;
}}

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
