# -*- coding: utf-8 -*-
"""台股判斷 · 基準日 2026-10-02。

**本日主軸：等權 +1.08%、中位 +0.17%、25/50 收紅 —— 平均遠高於中位，是少數載板／塑化股大漲拉高，不是普漲。**

**籌碼面本日回到 50/50 有值**（prepare 在 22:00 後跑，融資已上架；9/24 起多日為 0/50）。
p 的構成仍與 9/30、10/1 相同（成長排序 ±0.03、便宜度 ±0.015），**籌碼不進 p**：
本系統唯一經 PIT 檢驗 |t|>2 的橫斷面訊號是營收年增（LESSONS 2026-09-19），
籌碼訊號沒有驗過，有值不等於有效。金控 12 檔仍棄權，但理由改寫（不再是「缺資料」）。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.quantiles import quantiles

AS_OF = "20261002"
ANCHOR = 0.53   # METHOD §4.3；台股 h5 已結算有效天數約 1.2，不據以調錨（LESSONS G1）

FIN = {"2880", "2881", "2882", "2883", "2884", "2885", "2886", "2887",
       "2890", "2891", "2892", "5880"}
MEMORY = set()   # 美光 9/30 盤後財報的台股反應已發生在本日報酬內，事件已落地，不再棄權
# 本益比分母在循環谷底或剛轉上來：不以 PER 扣分（沿用 9/23 的處置）
PER_TROUGH = {"3189", "8046", "3653"}
# 本益比分母在循環高點：不以 PER 加分
PER_PEAK = {"2408", "2344"}

MACRO = (
    "基準日 10/2：本系統資料為等權 +1.08%、中位 +0.17%、25/50 檔收紅；平均比中位高近 0.9 個百分點，是少數大漲檔拉高，寬度只有一半。\n\n"
    "**領漲**：臻鼎 +10.0%（全場最高；新聞：AI 高階 PCB 與載板題材、當日外資與投信同列買超名單）、"
    "欣興 +7.4%（第 2；20 日 +34.1% 為全場最高；新聞：買下普利司通新竹廠擴產、載板報價調漲）、"
    "台塑化 +6.2%（第 3；投信 5 日買超 0.164 為全場最高；新聞：油價上揚與花旗、瑞銀看多塑化）。\n\n"
    "**領跌**：緯創 −2.1%（全場最低；新聞：外資賣超主動式 ETF 持股名單含緯創）、奇鋐 −2.0%、台新新光金 −1.7%，跌幅都不大。\n\n"
    "**5 日**：南電 +20.7% 為全場最高、臻鼎 +18.6% 第 2、景碩 +15.9% 第 3 —— ABF 載板三檔包辦前三；鴻勁 −6.6% 為全場最低。"
    "**20 日最低**：大立光 −19.5%（營收近三月累計 −12.4%，非金控唯一負值）。**本益比最高**：創意 202.5。\n\n"
    "**籌碼面本日 50/50 有值**（foreign_20 49/50）。本批**不以籌碼計分**：籌碼訊號在本系統沒有經過 PIT 的 IC 檢驗，"
    "營收年增是唯一 |t|>2 的訊號。也不引用融資的極值 —— 第一金 margin_chg_20 再度居首，但 LESSONS 2026-09-19 的基數判準本日未逐檔重跑。\n\n"
    "**事件視窗**：50 檔的 20 日視窗內皆有 9 月營收公告（6 個營業日後），台積電另有季報；5 日視窗內無事件。\n\n"
    "**一年期**：新聞閘本日標記光寶（88.5 億擴建高雄電源廠）、欣興（買普利司通新竹廠擴產）、日月光（海外購地與設備採購）三則擴產新聞。"
    "已讀：欣興的已實查論點本來就寫「載板是擴產競賽產業」，這則新聞是在印證前提而不是推翻它；光寶、日月光為規則推導，擴產不改動任何檢查點。三檔論點不變。\n\n"
    "**p 的主導維度是基本面**。**刻意不做反轉**：不因載板股連漲或鴻勁、致茂 5 日急跌加減分。"
    "**錨點維持 0.53**：台股已結算批次的有效天數約 1.2，不調錨（LESSONS G1）。"
)

# 手動覆寫：code → (p 覆寫或 None, inference 補充)
NOTE = {
    "3189": (0.535, "規則值落在錨點之下，是因近三月累計成長在排序後段；"
             "但近期新聞具名催化劑（ABF 載板缺貨、漲價）是規則看不到的前瞻資訊，"
             "故拉到錨點上方一檔（沿用 9/24 起）。本益比 194.8 是循環谷底分母，不扣分。"),
    "8046": (None, "與景碩同屬 ABF 載板，缺貨漲價敘事相同；5 日 +20.7% 為全場最高，不因延伸扣分。本益比為谷底分母，不扣分。"),
    "4958": (None, "本日 +10.0% 為全場最高、5 日 +18.6% 為全場第 2；外資 5 日買超 0.126。新聞為 AI 高階板與載板題材、目標價上調，無可對帳的新數字。不追漲，由規則計分。"),
    "3037": (None, "20 日 +34.1% 為全場最高、RSI 74.7 亦為全場最高；新聞為買廠擴產與載板報價。不因延伸扣分（反轉 IC 為負），由規則照常計分。"),
    "3443": (None, "20 日 +34.0% 為全場第 2；本益比 202.5 為全場最貴，只由規則的便宜度項扣分，不因延伸另扣。"),
    "6505": (None, "本日 +6.2% 為全場第 3、投信 5 日買超 0.164 為全場最高；新聞為油價與外資升評。籌碼不進 p，由規則計分。"),
    "6446": (None, "20 日 −10.7% 而營收近三月累計 +97.7% 在前段，是價格與基本面背離；距 60 日高點 −26.5% 為全場第 2 深。"),
    "7769": (0.53, "營收欄缺值，成長項無從計分；5 日 −6.6% 與 RSI 40.0 皆為全場最低，但不做反轉。維持錨點棄權（沿用 9/29）。"),
    "3008": (None, "20 日 −19.5% 為全場最低；營收近三月累計 −12.4%，規則自然落在低端。不另加減。"),
    "2327": (None, "距 60 日高點 −30.4% 為全場最深（7 月起崩跌是真實價格，分割在 2025-08-25，LESSONS A2）；"
             "近日外資大買的新聞無可對帳數字。由規則計分。"),
    "2454": (0.53, "規則值偏低是因近三月累計年增 +18.4% 落後單月 +44.1%，"
             "三月累計是落後指標，拿它懲罰一個正在加速的營收等於賭它不轉。故收回錨點棄權（沿用 9/24 起）。"),
    "2360": (None, "本日 +5.3%，但 5 日 −5.8% 為全場第 2 低；投信 5 日賣超 −0.232 為全場最強、外資 5 日買超 0.164 方向相反。"
             "營收近三月累計 +131.8% 在前段。籌碼互相抵銷且不進 p，由規則計分。"),
    "3231": (None, "本日 −2.1% 為全場最低；新聞為外資賣超主動式 ETF 的持股。營收近三月累計 +90.1%、本益比 14.8 都在有利端，由規則計分。"),
    "2345": (None, "20 日 −8.2% 而營收近三月累計 +63.9%；投信 5 日賣超 −0.187 為全場第 2 強。籌碼不進 p，由規則計分。"),
}

FIN_INF = ("本日籌碼 50/50 有值，但籌碼訊號在本系統沒有經過 PIT 檢驗、IC 未知；"
           "營收年增對金控是投資收益的波動而非本業成長，不可與製造業同尺度排序。"
           "唯一有證據的訊號用不上、能用的訊號沒有證據，故本檔棄權，p＝錨點，不計入命中率。")
MEM_INF = (""
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
            fal = (f"棄權檔。若 {name} 5 日內單邊變動逾 {thr*100/2:.0f}%，"
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
                      "當日等權 +1.08%、中位 +0.17%、25/50 檔收紅（由 briefing 計算）",
                      "籌碼面 foreign_5／trust_5／margin_chg_5／short_ratio 本日 50/50 有值，但不進 p"],
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
