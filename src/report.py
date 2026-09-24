"""產生靜態 HTML 儀表板（可直接部署 GitHub Pages）。"""
from __future__ import annotations
import datetime as dt, html, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, DOCS, PREDICTIONS, SETTLEMENTS
import score as score_mod
import ranking as ranking_mod
from collect import universe


def _jsonl(p: Path) -> pd.DataFrame:
    if not p.exists(): return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def _tbl(df: pd.DataFrame, cls: str = "") -> str:
    if df.empty: return '<p class="muted">（尚無資料）</p>'
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in r) + "</tr>"
                for r in df.itertuples(index=False))
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def build() -> Path:
    preds = _jsonl(PREDICTIONS)
    setts = _jsonl(SETTLEMENTS)
    uni = universe.load()
    names = {c["code"]: c["name"] for c in uni["constituents"]}

    latest = preds["as_of"].max() if not preds.empty else "—"
    today = preds[preds["as_of"] == latest] if not preds.empty else pd.DataFrame()

    # 期望值排序（主體）
    board = ""
    for h in (20, 5):
        t = ranking_mod.table(h)
        if t.empty:
            continue
        v = t[["code", "名稱", "prob_up", "up_magnitude", "dn_magnitude", "exp_ret",
               "reward_risk", "asymmetry", "conviction", "rsi_14", "rev_yoy",
               "ret_20", "foreign_5", "rationale"]].copy()
        v["prob_up"] = (v["prob_up"] * 100).round(0).astype(int).astype(str) + "%"
        for c, d in (("up_magnitude", 1), ("dn_magnitude", 1), ("exp_ret", 2),
                     ("rev_yoy", 1), ("ret_20", 1)):
            v[c] = v[c].apply(lambda x: "n/a" if pd.isna(x) else f"{x*100:+.{d}f}%")
        v["foreign_5"] = v["foreign_5"].apply(lambda x: "n/a" if pd.isna(x) else f"{x:+.2f}")
        v["rsi_14"] = v["rsi_14"].round(0).astype(int)
        v["rationale"] = v["rationale"].astype(str).str.split(": ", n=1).str[-1]
        v.insert(0, "#", range(1, len(v) + 1))
        v.columns = ["#", "代號", "名稱", "P(漲)", "漲幅", "跌幅", "期望值", "賠率比",
                     "報酬風險", "信心", "RSI", "營收YoY", "20日", "外資5日", "判斷理由"]
        pos = t[t["exp_ret"] > 0]
        board += (f"<h3>未來 {h} 個交易日 · 正期望值 {len(pos)} 檔 / 全體均值 "
                  f"{t['exp_ret'].mean()*100:+.2f}%</h3>" + _tbl(v, "num"))

    # 成績單
    sc = score_mod.summary()
    grade = ""
    if sc.get("rows"):
        for h, d in sc["by_horizon"].items():
            # 用 .get()：成績單的列有兩種（彙總與分版本），schema 已統一，
            # 但下游不該因為上游多一種列就整份報表掛掉。
            rows = [{"模型": m["model"], "筆數": m["n"],
                     "準確率": ("—" if m.get("accuracy") is None else f"{m['accuracy']:.1%}"),
                     "Brier": m.get("brier"),
                     "技能分數": ("—" if m.get("brier_skill") is None
                                else f"{m['brier_skill']:+.3f}"),
                     "對猜漲超額": "—" if m.get("edge_vs_always_up") is None
                                  else f"{m['edge_vs_always_up']:+.1%}"}
                    for m in d["models"]]
            grade += (f"<h3>期間 {h} 日 · 實際上漲率 {d['base_rate_up']:.1%}</h3>"
                      + _tbl(pd.DataFrame(rows), "num"))
    else:
        n_pend = len(preds) if not preds.empty else 0
        grade = (f'<p class="muted">尚無已結算的預測。目前有 <b>{n_pend}</b> 筆等待到期；'
                 '最快的 5 日期間預測將在一週後產生第一批成績。</p>')

    # 回測（含重疊調整）
    bt = ""
    cs_p = DATA / "backtest/crosssec.parquet"
    if cs_p.exists():
        cs = pd.read_parquet(cs_p)
        for h, g in cs.groupby("horizon"):
            g2 = g[["model", "rank_ic", "ic_hit_rate", "spread", "t_stat", "n_eff", "t_adj"]].copy()
            g2.columns = ["模型", "RankIC", "IC>0比例", "前20-後20", "名目t", "有效n", "調整t"]
            g2["IC>0比例"] = (g2["IC>0比例"] * 100).round(0).astype(int).astype(str) + "%"
            g2["前20-後20"] = (g2["前20-後20"] * 100).round(2).astype(str) + "%"
            bt += f"<h3>期間 {h} 日</h3>" + _tbl(g2, "num")

    gen = dt.datetime.now(dt.UTC).astimezone(dt.timezone(dt.timedelta(hours=8)))
    n_models = today["model"].nunique() if not today.empty else 0

    css = """
:root{--bg:#fbfaf8;--fg:#1b1a18;--mut:#6b6862;--line:#e5e1da;--card:#fff;
--ok:#2f6f4e;--bad:#a4453a;--accent:#2b5c8a}
@media(prefers-color-scheme:dark){:root{--bg:#141413;--fg:#eceae5;--mut:#96918a;
--line:#2c2a27;--card:#1c1b19;--ok:#6fbf8e;--bad:#e0847a;--accent:#7fb0d9}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans TC",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:38px 0 12px;
padding-bottom:8px;border-bottom:2px solid var(--line)}
h3{font-size:15px;margin:22px 0 8px;color:var(--mut);font-weight:600}
.muted{color:var(--mut)}.sub{color:var(--mut);font-size:13px;margin:0 0 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .k{font-size:12px;color:var(--mut)}.card .v{font-size:22px;font-weight:650;margin-top:2px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card);
border:1px solid var(--line);border-radius:8px;overflow:hidden}
th,td{padding:7px 11px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:color-mix(in srgb,var(--card) 88%,var(--fg));font-weight:600;font-size:12.5px}
tbody tr:last-child td{border-bottom:none}
table.num td:not(:nth-child(-n+2)),table.num th:not(:nth-child(-n+2)){text-align:right;
font-variant-numeric:tabular-nums}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
border-radius:8px;padding:14px 18px;margin:18px 0;font-size:14px}
.note b{color:var(--accent)}
footer{margin-top:50px;padding-top:18px;border-top:1px solid var(--line);
color:var(--mut);font-size:12.5px}
"""
    doc = f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>股市預測驗證系統</title><style>{css}</style></head><body><div class="wrap">
<h1>股市預測驗證系統</h1>
<p class="sub">母體：市值前 {uni['size']} 大上市公司 · 資料截止 {latest} · 產生於 {gen:%Y-%m-%d %H:%M} (台北)</p>

<div class="cards">
<div class="card"><div class="k">累積預測</div><div class="v">{len(preds):,}</div></div>
<div class="card"><div class="k">已結算</div><div class="v">{len(setts):,}</div></div>
<div class="card"><div class="k">今日出手模型</div><div class="v">{n_models}</div></div>
<div class="card"><div class="k">追蹤標的</div><div class="v">{uni['size']}</div></div>
</div>

<div class="note"><b>這個系統的用途不是賺錢，是防止自我欺騙。</b>
每日預測一旦寫入就不可修改，並與「永遠猜漲」等笨基準線並排計分。
任何模型若贏不過基準線，它的價值就是零 —— 這件事只有事前把預測鎖死才驗證得了。</div>

<h2>期望值排序</h2>
<div class="note"><b>期望值 = P(漲) × 上漲時幅度 + P(跌) × 下跌時幅度。</b>
只看方向機率會漏掉幅度：一檔 P(漲)=0.6 但漲 1%／跌 3% 的標的，期望值其實是負的。
「賠率比」= |漲幅／跌幅|，偏離 1 代表報酬與風險不對稱，已於「報酬風險」欄標註。</div>
{board or '<p class="muted">（今日尚未產生判斷）</p>'}

<h2>實際成績單（前瞻，唯一可信的分數）</h2>
<div class="note">基準線（always_up／random／momentum）不是「另一個模型」，而是<b>尺規</b>。
判斷若贏不過「無條件猜漲」，它的價值就是零 —— 沒有這把尺，三個月後無從得知判斷到底有沒有用。</div>
{grade}

<h2>歷史回測（僅供參考，不等於未來）</h2>
<div class="note">回測有三個無法消除的限制：<b>①</b> LLM 知道歷史結果，完全無法回測；
<b>②</b> 特徵與模型是事後挑選的，這個研究者自由度不會因 walk-forward 而消失；
<b>③</b> 相鄰樣本重疊會高估顯著性 —— 請一律看「調整t」，|t|&lt;2 代表沒有證據。</div>
{bt or '<p class="muted">（尚未執行回測）</p>'}

<footer>資料來源：FinMind（價量／法人／融資券）、TWSE OpenAPI（市值）、Google News。
本頁為研究與紀律工具，不構成投資建議。</footer>
</div></body></html>"""
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "report.html"   # 不可寫 index.html —— 那是 PWA 主頁，由 panel.py 產生
    out.write_text(doc, encoding="utf-8")
    return out


if __name__ == "__main__":
    p = build()
    print(f"報表已產生：{p}  ({p.stat().st_size/1024:.0f} KB)")
