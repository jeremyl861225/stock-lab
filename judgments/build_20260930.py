# -*- coding: utf-8 -*-
"""台股判斷 · 基準日 2026-09-30。

**本日的主軸：休市後第二天反彈，寬度由 17/50 回到 29/50；領漲是面板、塑化與 PCB／連接器。**

  等權 +0.87%、中位 +0.20%、29/50 收紅（由 panel 計算）。

**籌碼面 foreign_5／trust_5／margin_chg_5／short_ratio 本日 0/50 有值**
  （prepare 15:47 跑完，當日三大法人與融資尚未公布）。處置沿用 9/24、9/29：
  一律不引用籌碼，以籌碼為主要辨識維度的金控 12 檔棄權。

p 的構成與 9/29 相同（規則：近三月累計營收年增排序 ±0.03 ＋ 本益比便宜度 ±0.015，
少數檔手動覆寫並寫理由）；不對 20 日漲幅扣分（反轉 IC 為負，LESSONS 2026-09-19）。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.quantiles import quantiles

AS_OF = "20260930"
ANCHOR = 0.53   # METHOD §4.3；結算只有 0.4 個有效天數，不據以調錨（LESSONS G1）

FIN = {"2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887",
       "2890", "2891", "2892", "5880"}
MEMORY = {"2408", "2344"}
# 本益比分母在循環谷底或剛轉上來：不以 PER 扣分（沿用 9/23 的處置）
PER_TROUGH = {"3189", "8046", "3653"}
# 本益比分母在循環高點：不以 PER 加分
PER_PEAK = {"2408", "2344"}

MACRO = (
    "基準日 9/30：本系統資料為等權 +0.87%、中位 +0.20%、29/50 檔收紅（新聞：加權指數反彈 308 點）。\n\n"
    "**寬度回升**：上漲檔數 9/23 22/51 → 9/24 21/50 → 9/29 17/50 → 9/30 29/50。"
    "等權明顯高於中位，是少數大漲檔拉高平均，不是普漲。\n\n"
    "**領漲是面板、塑化、PCB／連接器**：群創 +9.9%（漲停；新聞：面板與玻璃基板新應用題材）、"
    "貿聯 +6.1%（盤中觸漲停；Interplex 併表、法人上調目標價）、南電 +5.5%（5 日 +26.4% 為全場最高；ABF 載板漲價敘事）、"
    "台塑 +5.4%、南亞 +3.4%、台塑化 +3.2%（外資升評、CCL 報價）、華邦電 +3.2%、台達電 +3.0%。\n\n"
    "**領跌幅度都小**：致茂 −2.3%（5 日 −10.1% 為全場最低；新聞為主動式 ETF 減碼，無具名基本面利空）、"
    "智邦 −1.9%（RSI 35.6 為全場最低；新聞：投信季底結帳、主動式 ETF 減碼）、健策 −1.4%、鴻勁 −1.4%。\n\n"
    "**延伸最多的**：台塑化 20 日 +32.4%、RSI 74.4，兩項皆為全場最高；創意 20 日 +31.8% 居次。"
    "**跌最深的**：大立光 20 日 −22.3% 為全場最低（營收近三月累計年減 −12.4%，非金控唯一負值）。\n\n"
    "**資料缺口：籌碼面 0/50**。本批不引用任何籌碼數字；金控 12 檔棄權。\n\n"
    "**記憶體棄權**：美光 FQ4 財報在美東 9/30 盤後（台股 10/1 反應），落在 5 日視窗內，"
    "是南亞科、華邦電的二元事件；兩檔營收年增是循環高點分母，不與其他成長股同尺度排序。\n\n"
    "**事件視窗**：50 檔的 20 日視窗內皆有 9 月營收公告（8 個營業日後）；台積電另有季報。5 日視窗內無月營收。\n\n"
    "**p 的主導維度是基本面**（成長排序 ±0.03、便宜度 ±0.015）。**刻意不做反轉**：不因今日漲停（群創）或 5 日急跌（致茂）"
    "加減分。**錨點維持 0.53**：台股 5 日已結算的有效天數仍不足 1（9/21 那批今日到期），依 LESSONS G1 不調錨。"
)

# 手動覆寫：code → (p 覆寫或 None, inference 補充)
NOTE = {
    "3481": (None, "本日 +9.9% 漲停，新聞為面板與玻璃基板新應用題材（題材，非營收數字）；營收年增 0.0%、"
             "近三月累計 +5.0% 在排序後段，規則值自然偏低。不追題材、不因漲停加分。"),
    "3653": (None, "RSI 70.5 為全場第 2（台塑化 74.4 較高）；新聞：8 月自結 EPS 7.35 元、法人上修 2027 年獲利。"
             "本益比高是分母剛轉上來，不扣分。"),
    "3189": (0.535, "規則值落在錨點之下，是因近三月累計成長在排序後段；"
             "但 9/24 起的新聞具名催化劑（ABF 載板缺貨、漲價）是規則看不到的前瞻資訊，"
             "故拉到錨點上方一檔（沿用 9/24、9/29）。本益比高是循環谷底分母，不扣分。"),
    "8046": (None, "與景碩同屬 ABF 載板，缺貨漲價敘事相同；5 日 +26.4% 為全場最高，本批不因延伸扣分。本益比為谷底分母，不扣分。"),
    "3443": (None, "20 日 +31.8% 為全場第 2；本益比 203.2 為全場最貴，只由規則的便宜度項扣分，不因延伸另扣。"),
    "6446": (None, "20 日 −12.9%（全場第 3 深）而營收成長在前段，是價格與基本面背離。"
             "9/28 完成收購予宇生技（約 1.68 億元，LNP 技術），相對公司規模小，不改論點。"),
    "7769": (0.53, "營收欄缺值，成長項無從計分；近期新聞為外資喊買與『注意股』公告，皆非可對帳的數字。"
             "故維持錨點棄權（沿用 9/29）。"),
    "2303": (None, "新聞為成熟製程報價調升與 ETF 換股（賣壓已發生），無具名新利空。"),
    "3008": (None, "20 日 −22.3% 為全場最深；營收近三月累計年減 −12.4%，規則自然落在低端。"
             "新聞提到 9 月拉貨動能與 10 月法說（CPO 題材），屬前瞻敘述、無數字，不另加減。"),
    "2327": (None, "7 月起的崩跌是真實價格（分割在 2025-08-25，近 60 日 adj_factor 恆為 1.0，LESSONS A2）；"
             "距 60 日高點 −45.5% 為全場最深。"),
    "2454": (0.53, "規則值偏低是因近三月累計年增 +18.4% 落後單月 +44.1%，"
             "三月累計是落後指標，拿它懲罰一個正在加速的營收等於賭它不轉。故收回錨點棄權（沿用 9/24、9/29）。"),
    "3665": (None, "Interplex 資料通訊部門 9/23 起併表、法人上調獲利預估；本日盤中觸及漲停、收 +6.1%。"
             "規則值未納入併表，下一季財報自然反映。"),
    "2360": (None, "5 日 −10.1% 為全場最低，新聞為主動式 ETF 經理人減碼（資金面），無具名基本面利空；"
             "營收近三月累計 +131.8% 在前段。不因急跌加分（不做反轉），由規則照常計分。"),
    "2345": (None, "RSI 35.6 為全場最低、距 60 日高點 −32.4%；新聞為投信季底結帳與主動式 ETF 減碼，"
             "同期另有 8 月營收創新高的報導。超賣不加分，由規則照常計分。"),
    "6505": (None, "RSI 74.4、20 日 +32.4% 兩項皆為全場最高；新聞為油價、外資升評與目標價上調。"
             "不因延伸扣分（反轉 IC 為負），由規則照常計分。"),
}

FIN_INF = ("金控的 5／20 日辨識維度是籌碼擁擠度（融資、外資），本日 0/50 缺值；"
           "營收年增對金控是投資收益的波動而非本業成長，不可與製造業同尺度排序。"
           "故本檔棄權，p＝錨點，不計入命中率。")
MEM_INF = ("美光財報美東 9/30 盤後（台股 10/1 反應）落在 5 日視窗內，是記憶體族群的二元事件；"
           "營收年增數百個百分點是循環高點的分母，不能與其他成長股同尺度排序。故棄權，p＝錨點。")


def pct(x):
    return f"{x*100:+.1f}%"


def rank_desc(s: pd.Series, code: str) -> int:
    s = s.dropna().sort_values(ascending=False)
    return list(s.index).index(code) + 1


def make(b: pd.DataFrame):
    b = b.set_index("code")
    nf = b[~b.index.isin(FIN)]
    g = nf["rev_yoy3m"].rank(pct=True)
    per = nf["PER"].copy()
    v = (-per).rank(pct=True)
    J = []
    for code, r in b.iterrows():
        name = r["名稱"]
        facts = (f"{name}：本日 {pct(r['ret_1'])}、5 日 {pct(r['ret_5'])}、20 日 {pct(r['ret_20'])}、"
                 f"距 60 日高點 {pct(r['dist_high_60'])}、RSI {r['rsi_14']:.1f}、"
                 f"日波動 {r['vol_20']*100:.1f}%。")
        if pd.notna(r["rev_yoy"]):
            facts += (f"營收年增 {pct(r['rev_yoy'])}（近三月累計 {pct(r['rev_yoy3m'])}，"
                      f"非金控有值 {nf['rev_yoy3m'].notna().sum()} 檔中第 {rank_desc(nf['rev_yoy3m'], code) if code in nf.index else '—'} 高）、")
        else:
            facts += "營收欄缺值、"
        facts += f"本益比 {r['PER']:.2f}、殖利率 {r['dividend_yield']:.2f}%。"
        if bool(r.get("evt_in_20")):
            facts += (f"20 日視窗內事件：{r['evt_types_20']}"
                      f"（最近一件 {r['evt_type']}，{int(r['evt_days'])} 個營業日後）。")
            if "財報" in str(r["evt_types_20"]):
                facts += "20 日判斷含財報、5 日判斷不含，兩者是不同的賭局。"
        sig20 = 0.8 * r["vol_20"] * math.sqrt(20)
        thr = max(0.04, round(sig20, 2))
        if code in FIN:
            p, skew, inf = ANCHOR, 0.0, FIN_INF
            fal = (f"棄權檔。若 {name} 20 日內絕對報酬逾 {thr*100:.0f}%（約 1σ），"
                   "代表缺籌碼資料時棄權漏掉了方向性訊息，下次應以營收或殖利率補位。")
        elif code in MEMORY:
            p, skew, inf = ANCHOR, 0.0, MEM_INF
            fal = (f"棄權檔。若美光 9/30 財報後 {name} 5 日內單邊變動逾 {thr*100/2:.0f}%，"
                   "代表事件方向其實可以事前判斷（例如由報價新聞），棄權是錯的。")
        else:
            gp = g.get(code, 0.5) if pd.notna(g.get(code, np.nan)) else 0.5
            vp = v.get(code, 0.5)
            if code in PER_TROUGH or code in PER_PEAK:
                vp = 0.5
            p = ANCHOR + 0.03 * (2 * gp - 1) + 0.015 * (2 * vp - 1)
            rule_p = round(min(0.57, max(0.49, p)), 3)
            ov, extra = NOTE.get(code, (None, ""))
            p = rule_p if ov is None else ov
            skew = round((p - ANCHOR) * 2, 3)
            inf = (f"規則起算：成長百分位 {gp:.2f}、便宜度百分位 {vp:.2f}"
                   f"{'（本益比分母失真，便宜度取中性）' if code in PER_TROUGH | PER_PEAK else ''}"
                   f" → 規則值 {rule_p:.3f}" + (f"，手動覆寫為 {p:.3f}（理由見下）。" if ov is not None else "。"))
            if ov is None:
                inf += ("看多的理由是營收成長在排序前段而估值未墊高到抵銷。" if p > ANCHOR else
                        "看空的理由是營收成長落後或估值偏貴，而非股價延伸。" if p < ANCHOR else
                        "規則落在錨點。")
            if p > ANCHOR:
                fal = (f"若 {name} 20 日內下跌逾 {thr*100:.0f}%（約 1σ），"
                       f"代表本檔的看多理由無效；5 日內下跌逾 {thr*50:.0f}% 為早期警訊。")
            elif p < ANCHOR:
                fal = (f"若 {name} 20 日內上漲逾 {thr*100:.0f}%（約 1σ），"
                       "代表本檔的看空理由在這段多頭中不構成壓力。")
            else:
                fal = (f"棄權檔。若 {name} 20 日內絕對報酬逾 {thr*100:.0f}%（約 1σ），"
                       "代表收回錨點的理由錯了、規則值本來是對的。")
            if extra:
                inf += extra
        J.append((code, p, skew, "low", facts, inf, fal))
    return J


def build(J, horizon: int, vol: dict) -> dict:
    out = []
    for code, p20, skew, conf, thesis, inf, fal in J:
        p = 0.5 + (p20 - 0.5) * math.sqrt(horizon / 20)
        base = 0.85 * vol.get(code, 0.02) * math.sqrt(horizon)
        up, dn = base * (1 + skew), -base * (1 - skew)
        ev = p * up + (1 - p) * dn
        sig = vol.get(code, 0.02) * math.sqrt(horizon)
        q10, q90 = quantiles(ev, sig)
        ev = round(ev, 5)
        ratio = abs(up / dn) if dn else float("inf")
        tag = "正偏（上檔大）" if ratio > 1.25 else ("負偏（下檔大）" if ratio < 0.8 else "對稱")
        out.append({
            "code": code, "stance": "bullish" if ev >= 0 else "bearish",
            "conviction": conf, "prob_up": round(p, 3), "exp_ret": ev,
            "ret_q10": round(q10, 5), "ret_q90": round(q90, 5),
            "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
            "reward_risk": round(ratio, 2), "asymmetry": tag,
            "thesis": thesis, "inference": inf,
            "facts": [thesis,
                      "當日等權 +0.87%、中位 +0.20%、29/50 檔收紅（由 panel 計算）",
                      "籌碼面 foreign_5／trust_5／margin_chg_5／short_ratio 本日 0/50 有值"],
            "falsifier": fal,
        })
    return {"as_of": AS_OF, "horizon": horizon, "analyst": "claude-opus-5",
            "market": "TW", "version": "v1", "anchor": ANCHOR,
            "market_context": MACRO, "judgments": out}


if __name__ == "__main__":
    b = pd.read_parquet("data/briefing.parquet")
    b = b[b["market"] == "TW"]
    assert b["as_of"].astype(str).str.replace("-", "").str[:8].eq(AS_OF).all(), "briefing 不是本日資料"
    vol = dict(zip(b["code"], b["vol_20"].fillna(0.02)))
    J = make(b)
    codes = [c for c, *_ in J]
    assert len(codes) == len(set(codes)) == 50
    ps = pd.Series([x[1] for x in J])
    print(f"  p 標準差 {ps.std():.4f}、範圍 {ps.min():.3f}–{ps.max():.3f}、"
          f"高於錨點 {(ps > ANCHOR).sum()}、等於 {(ps == ANCHOR).sum()}、低於 {(ps < ANCHOR).sum()}")
    for h, suffix in ((20, ""), (5, "_h5")):
        d = build(J, h, vol)
        p = Path("judgments") / f"{AS_OF}{suffix}.json"
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        ev = [x["exp_ret"] for x in d["judgments"]]
        print(f"  h={h:<3} {len(d['judgments'])} 檔 → {p.name}  "
              f"期望值 {min(ev)*100:+.2f}% ~ {max(ev)*100:+.2f}%")
