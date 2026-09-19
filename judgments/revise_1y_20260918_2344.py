# -*- coding: utf-8 -*-
"""一年期論點修訂 · 2344 華邦電 · 2026-09-18 基準日。

**這份修訂是新聞閘觸發的，但真正的理由不只是那則新聞。**

觸發：2026-09-16～18 三天，新聞閘在 943 則標題裡命中 2344 的併購事件
（`src/news_gate.py`，分類「併購分拆」）。依 DAILY.md 第一點七節，
我去讀了那幾則，得到兩個結論，第二個比第一個重要。

一、併購本身在一年期窗口內幾乎不影響財報。
    交易預計 **2027 年下半年**完成，而本預測的窗口是 2026Q4–2027Q3。
    窗口內不會有任何合併綜效或營收認列，只有融資安排會先上資產負債表。
    所以我沒有因為併購而調高成長，那會是把窗口外的事算進窗口內。

二、讀這檔的時候發現，規則模型把它排到台股一年期期望值的最前段
    （P漲 0.621、期望值 +23.4%），依據是毛利率較四季前 **+43.6pp**、
    EPS TTM 年增 **+1232%**、本益比 19.8「便宜」。
    而同一天我對美光（MU）寫下的論點，講的正是這組數字的反面：
    「記憶體的歷史教訓是本益比最低的時候最貴 —— 分母是週期高點的獲利」，
    因此給 P漲 0.48。

    兩檔在同一個記憶體循環裡，一檔被判到最前段、一檔被判到最後段，
    差別只在**一個有人讀過、一個沒有**。那不是判斷的差異，是覆蓋率的差異。
    橫斷面百分位對「週期高點」沒有免疫力：它看到的是第 98、第 100 百分位，
    看不到那是循環的哪個位置。

    華邦電的毛利率長期在 20–35% 區間，66.2% 遠高於任何一次前高。
    要維持它，NOR Flash 與利基型 DRAM 的報價得停在現在的位置一整年。

修訂內容：P漲 0.621 → 0.50，信心 low → medium（併購已實查，
但仍未實查公司自身的財測與產能計畫，故不標 high），
論點改寫並補上一條資產負債表檢查點。

舊的那一筆不會被刪掉 —— ingest 以 revision 遞增並存（憲法 B）。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models.quantiles import quantiles

CODE, AS_OF = "2344", "20260918"
P_UP, SKEW, ANCHOR = 0.50, -0.03, 0.55

THESIS = "併購落在窗口之外，窗口之內要回答的是 66.2% 的毛利率能不能停在原地一整年"

FACTS = [
 "2026-09-16 重訊：以全現金 11.2 億美元（約新台幣 356 億元）取得英飛凌 NOR Flash "
 "與 F-RAM 事業 100% 股權，標的更名 Spansion，總部在美國加州聖荷西（公司重訊）",
 "**交易預計 2027 年下半年完成**，落在本預測窗口（2026Q4–2027Q3）之外（公司重訊）",
 "完成後全球 NOR Flash 市占率超過三成（媒體引述產業估計，非公司指引）",
 "2026Q2 毛利率 66.2%、較四季前 +43.6pp；營益率 48.4%；EPS TTM 9.06 元、年增 +1232%（財報）",
 "負債比 40.8%、ROE 28.5%、本益比 19.8、股價淨值比 4.9（財報與市場價格）",
 "華邦電歷史毛利率長期落在 20–35% 區間（歷史財報，本次為區間外的極值）",
]

INFERENCE = (
 "併購是好事但不在窗口內：2027 下半年才完成，窗口內只會看到融資安排上資產負債表，"
 "看不到任何綜效。把窗口外的成長算進窗口內，是一年期判斷最常見的錯法之一。\n\n"
 "窗口內真正的問題是毛利率。66.2% 比這家公司任何一次循環高點都高出約 30 個百分點，"
 "而本益比 19.8 的分母正是這個毛利率撐出來的 EPS —— 記憶體股在週期高點的本益比"
 "永遠是最低的，那不是便宜，是分母到頂。全現金 11.2 億美元的支付承諾"
 "又會在窗口內先產生融資需求，資產負債表的緩衝因此變薄。\n\n"
 "原判斷的 P漲 0.621 由橫斷面百分位推導（毛利率變化第 98、EPS 年增第 100 百分位），"
 "而百分位對「這是循環的哪個位置」沒有辨識力。同一天我對美光寫的論點"
 "（P漲 0.48）講的正是這組數字的反面。兩檔同屬一個循環卻被判到兩端，"
 "差別只在覆蓋率不在判斷 —— 這裡改成一致。不宣稱方向。"
)

CPS = [
 {"claim": "毛利率 ≥45%（報價回落但未回到循環中位）", "metric": "gross_margin",
  "op": ">=", "threshold": 0.45},
 {"claim": "EPS TTM 年增 ≥−30%（高基期下的回落未失控）", "metric": "eps_yoy",
  "op": ">=", "threshold": -0.30},
 {"claim": "月營收 TTM 年增維持正值", "metric": "rev_yoy_ttm", "op": ">=", "threshold": 0.0},
 # 全現金 11.2 億美元的支付承諾要在窗口內先安排融資。
 # 這一條是這次併購在窗口內**唯一**看得到的財報後果，所以必須有人盯。
 {"claim": "負債比 ≤55%（併購融資未把資產負債表撐爆）", "metric": "debt_ratio",
  "op": "<=", "threshold": 0.55},
]


def main() -> None:
    import pandas as pd
    import checkpoints as CP

    src = ROOT / "judgments" / f"{AS_OF}_1y.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    j = next(x for x in d["judgments"] if x["code"] == CODE)

    # 幅度用與建置時同一條式子重算，不沿用舊值 ——
    # P漲 改了，偏度與期望值都要跟著走。
    base = (j["up_magnitude"] - j["dn_magnitude"]) / 2      # 反解原 base
    up, dn = base * (1 + SKEW), -base * (1 - SKEW)
    ev = P_UP * up + (1 - P_UP) * dn
    sig = (j["up_magnitude"] - j["dn_magnitude"]) / 2 / 0.85
    q10, q90 = quantiles(ev, sig)

    j.update({
        "prob_up": round(P_UP, 4),
        "stance": "bullish" if ev >= 0 else "bearish",
        "thesis": THESIS, "facts": FACTS, "inference": INFERENCE,
        "rationale": INFERENCE,
        "falsifier": "；".join(c["claim"] for c in CPS) + " —— 任一前提被推翻即論點動搖",
        "researched": True, "conviction": "medium",
        "checkpoints": CPS,
        "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
        "exp_ret": round(ev, 6),
        "ret_q10": round(q10, 6), "ret_q90": round(q90, 6),
        "thesis_as_of": AS_OF, "repriced_only": False,
        "roll_note": "新聞事件（併購分拆）觸發人工重讀，論點已重寫",
    })

    # 出貨前一樣要過 day-one 關卡 —— 新寫的門檻不能當天就已經被推翻。
    bad = [c for c in CP.score_all(j, pd.Timestamp("2026-09-18"), "TW")["checkpoints"]
           if c["status"] == "broken"]
    if bad:
        print("！新檢查點在基準日就被推翻，請改門檻：")
        for c in bad:
            print(f"   {c['claim']}　實際 {c['actual']:+.3f}")
        sys.exit(1)

    d["judgments"] = sorted(d["judgments"], key=lambda x: -x["exp_ret"])
    src.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    # 狀態檔也要改。滾動時 P漲 取自狀態檔而非判斷檔
    # （`p_up = pj.get("prob_up", j["prob_up"])`），只改判斷檔的話，
    # 明天第一次滾動就會把 0.621 原封不動放回來，修訂等於沒發生。
    st_p = ROOT / "data/roll_1y_state.json"
    st = json.loads(st_p.read_text(encoding="utf-8"))
    # 狀態檔有兩種格式：舊的是頂層 {as_of, stocks}（只有台股），
    # 新的是 {markets: {TW: {...}, US: {...}}}。改版當天兩種都可能在磁碟上，
    # 只認新的會讓這段靜默跳過 —— 而「靜默跳過」的後果是明天滾動
    # 把舊的 P漲 原封不動放回來，修訂等於沒發生過。
    stocks = (st["stocks"] if "stocks" in st
              else st.get("markets", {}).get("TW", {}).get("stocks", {}))
    node = stocks.get(CODE)
    if node:
        node["prob_up"] = round(P_UP, 4)
        node["skew"] = SKEW
        node["thesis_as_of"] = AS_OF
        node["broken"] = CP.score_all(j, pd.Timestamp("2026-09-18"), "TW")["broken"]
        st_p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"狀態檔已同步：{CODE} P漲 → {P_UP}")
    else:
        print(f"！狀態檔沒有 {CODE}，明天滾動會用判斷檔的值（本次相同，但要知道）")

    print(f"{CODE} 已修訂：P漲 0.621 → {P_UP}　期望值 {ev:+.2%}"
          f"　（原 +23.39%）　檢查點 {len(CPS)} 條")


if __name__ == "__main__":
    main()
