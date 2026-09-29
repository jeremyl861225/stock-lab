# -*- coding: utf-8 -*-
"""台股判斷 · 基準日 2026-09-29（9/25、9/28 休市後的第一個交易日）。

**本日的主軸：休市後第一天，AI 高價 ASIC 股領跌，寬度續退到 17/50。**

  等權 −0.72%、中位 −0.73%、17/50 收紅（由 panel 計算）。
  上漲檔數 9/22 24/51 → 9/23 22/51 → 9/24 21/50 → 9/29 17/50。

**籌碼面 foreign_5／trust_5／margin_chg_5／short_ratio 本日 0/50 有值**
  （prepare 14:47 跑完，當日三大法人與融資尚未公布）。處置沿用 9/24：
  一律不引用籌碼，以籌碼為主要辨識維度的金控 12 檔棄權。

p 的構成與 9/24 相同（規則：近三月累計營收年增排序 ±0.03 ＋ 本益比便宜度 ±0.015，
少數檔手動覆寫並寫理由）；不對 20 日漲幅扣分（反轉 IC 為負，LESSONS 2026-09-19）。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.quantiles import quantiles

AS_OF = "20260929"
ANCHOR = 0.53   # METHOD §4.3；結算只有 0.4 個有效天數，不據以調錨（LESSONS G1）

FIN = {"2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887",
       "2890", "2891", "2892", "5880"}
MEMORY = {"2408", "2344"}
# 本益比分母在循環谷底或剛轉上來：不以 PER 扣分（沿用 9/23 的處置）
PER_TROUGH = {"3189", "8046", "3653"}
# 本益比分母在循環高點：不以 PER 加分
PER_PEAK = {"2408", "2344"}

MACRO = (
    "基準日 9/29（9/25、9/28 休市，TWSE 當日行情表為空）：本系統資料為等權 −0.72%、中位 −0.73%、"
    "17/50 檔收紅。\n\n"
    "**寬度續退**：上漲檔數 9/22 24/51 → 9/23 22/51 → 9/24 21/50 → 9/29 17/50。\n\n"
    "**領跌是 AI 高價 ASIC 股**：聯發科 −7.1%（新聞：失守 5000 元）、創意 −7.0%、國巨 −5.5%、"
    "致茂 −4.8%、智邦 −4.5%、奇鋐 −4.4%、台達電 −3.9%。聯發科與創意是 20 日漲幅第 3 與第 2 高的兩檔"
    "（+23.2%、+31.8%），本日是延伸後的回吐；兩檔新聞皆無具名利空（聯發科另有 ASIC 訂單與外資上調目標價）。\n\n"
    "**領漲是傳產與個別事件**：台塑化 +8.1%（新聞：油價上揚、地緣政治、煉油利差；20 日 +32.1% 為全場最高）、"
    "臻鼎 +4.8%（零利率可轉債籌資擴 AI 產能）、貿聯 +4.5%（Interplex 資料通訊部門 9/23 起併表、外資上調目標價）、"
    "南亞 +3.8%、藥華藥 +3.5%。\n\n"
    "**資料缺口：籌碼面 0/50**。本批不引用任何籌碼數字；金控 12 檔棄權。\n\n"
    "**記憶體棄權**：美光 FQ4 財報排在美東 9/30 盤後（台股 10/1 反應），落在 5 日視窗內，"
    "是南亞科、華邦電的二元事件；兩檔營收年增是循環高點分母，不與其他成長股同尺度排序。\n\n"
    "**事件視窗**：50 檔的 20 日視窗內皆有 9 月營收公告（9 個營業日後）；台積電另有季報。"
    "5 日視窗內無月營收。\n\n"
    "**p 的主導維度是基本面**（成長排序 ±0.03、便宜度 ±0.015）。**刻意不做反轉**：今日領跌的聯發科、創意"
    "不因延伸或單日大跌加減分。**錨點維持 0.53**：已結算 5 日 359 筆（含所有模型），有效天數不足 1，依 LESSONS G1 不調錨。"
)

# 手動覆寫：code → (p 覆寫或 None, inference 補充)
NOTE = {
    "3653": (None, "新聞具名：法人以新均熱片規格調高目標價（9/29）。RSI 73.0 為全場最高、本日 −0.4%；"
             "本益比高是分母剛轉上來，不扣分。"),
    "3189": (0.535, "規則值落在錨點之下，是因近三月累計成長在排序後段；"
             "但 9/24 新聞具名催化劑（ABF 載板缺貨、外資看千元、8 月 EPS 大增）是規則看不到的前瞻資訊，"
             "故拉到錨點上方一檔（沿用 9/24）。本益比高是循環谷底分母，不扣分。"),
    "8046": (None, "與景碩同屬 ABF 載板，缺貨敘事相同；本益比為谷底分母，不扣分。"),
    "3443": (None, "本日 −7.0% 是 20 日 +31.8%（全場第 2）之後的回吐，新聞無具名利空；"
             "本批不以延伸或單日跌幅加減分（反轉 IC 為負）。本益比 218.5 為全場最貴，只由規則的便宜度項扣分。"),
    "6446": (None, "20 日 −22.7% 為全場最深、營收成長在前段，是本批最大的價格與基本面背離。"
             "新聞為 ET 新藥日美上市與小額收購 LNP 技術；0050 換股傳聞標題為問句，不作為事實引用。"),
    "7769": (0.53, "營收欄缺值，成長項無從計分；9/24 用以壓低 p 的『主動式 ETF 重砍』新聞本期已不在新聞檔，"
             "近期新聞轉為外資喊買，兩者皆非可對帳的數字。故收回錨點棄權。"),
    "2303": (None, "新聞為成熟製程報價調升與 00919 換股清倉（賣壓已發生），無具名新利空。"),
    "3008": (None, "營收近三月累計年減，規則自然落在低端；不另加減。"),
    "2327": (None, "7 月起的崩跌是真實價格（分割在 2025-08-25，近 60 日 adj_factor 恆為 1.0，LESSONS A2）。"
             "9/28 新聞標題稱『連 2 根漲停』與 panel 不符（9/23 +0.3%、9/24 +1.2%），不採用。"),
    "2454": (0.53, "規則值偏低是因近三月累計年增 +18.4% 落後單月 +44.1%，"
             "三月累計是落後指標，拿它懲罰一個正在加速的營收等於賭它不轉；"
             "且本日 −7.1% 之後，低 p 在結果上等於追空。故收回錨點棄權（沿用 9/24）。"),
    "3665": (None, "Interplex 資料通訊部門（TTM 營收約 3.92 億美元、法人估占 2027 營收約 11%）9/23 起併表；"
             "以現金支付，法人上調 EPS 預估。規則值未納入併表，下一季財報自然反映。"),
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
                      "當日等權 −0.72%、中位 −0.73%、17/50 檔收紅（由 panel 計算）",
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
