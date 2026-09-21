# -*- coding: utf-8 -*-
"""一年期論點修訂 · 2454 聯發科 · 2026-09-21 基準日。

**這次修的主要是「前提的寫法」，不是方向。**

觸發：`prepare` 標記 2454「失效　已推翻 2 條」。去看被推翻的那兩條，
兩條都是我自己寫壞的 —— 與 2026-09-19 那次（VZ／QCOM／META）同一型：

  · 「EPS TTM 年增轉正（ASIC 確實接上）」—— 論點本文就寫著
    「財報還沒轉（EPS TTM 年增 −9.2%）」。**賭它翻過來正是論點的內容**，
    它在寫下當天本來就是假的。這是 checkpoints.py 定義的
    expects="turn"（未達成），不是 broken。
  · 「月營收 TTM 年增 ≥15%（量產未遞延）」—— 第一顆 ASIC 2026Q4 才量產，
    TTM 要到放量後幾季才追得上。門檻 15% 寫在當時現值（約 4–8%）之上，
    同樣是「把一個預測寫成了前提」。

LESSONS 2026-09-19 已經記過這件事，並說「台股 2454 就是這一型」，
還說 checkpoints 加了 expects="turn"。但實際翻過四份判斷檔：
**111 條檢查點裡標了 expects 的是 0 條** —— 機制寫好了，一條都沒用上。
於是這檔從第一天起就掛著「失效」，而這個機制的全部價值在於
「亮起來的時候一定有事」。摻了假警報等於把它關掉。

**同一天另外查到的（已修 src/checkpoints.py）**：`_monthly_rev` 的可用日
比法定公告日晚整整一個月，所有 rev_yoy／rev_yoy_ttm 檢查點都在用
上上個月的營收對帳。修正後 3037／3008／1326 三檔的「動搖」直接消失
（那三條前提根本沒倒）。2454 這兩條與那個 bug 無關，是真的寫壞了。

**P漲 不動，維持 0.52。** 這次沒有出現足以改變方向的新資訊，兩邊拉扯而且幅度相當：

  · 偏多：8 月營收年增 **+44.1%**，是這一串裡最陡的一次加速
    （4 月 −4.1%、5 月 +5.0%、6 月 +2.8%、7 月 +12.2%、8 月 +44.1%）。
    轉機論點賭的那件事開始出現在月營收上了。
  · 偏空：本益比由論點寫下時的 74.8 升到 **82.74**（三個交易日 +10.6%），
    而 EPS TTM 年增仍是 −9.2% —— 分子沒動，倍數自己漲上去。
    論點的核心反對意見「74.8 倍沒有留下毛利率稀釋的空間」在 82.7 倍只會更成立。

兩者方向相反、量級相當，所以維持 0.52（一年期中性錨 0.55 之下一格）。
把 P漲 為了「有在更新」而動一動，就是 DAILY.md 警告的把雜訊包裝成更新。
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models.price_1y import price as price_1y

CODE, AS_OF = "2454", "20260921"
# 一年期**沒有獨立的 skew**：對數常態的不對稱完全由 σ 決定（models/price_1y.py，
# 2026-09-19 改版）。judgments/revise_1y_20260918_2344.py 那份範本還停在舊的
# 兩點模型（base×(1±skew)、ev=p·up+(1−p)·dn），照抄會把 exp_ret 從 +3.08%
# 悄悄改成 +1.70% —— P漲 明明沒動。改用與 roll_1y 同一條式子，
# P漲 不變時導出的五個價格欄位就會與滾動產出的完全一致。
P_UP = 0.52                     # 維持原值，只改前提的寫法與事實

THESIS = "ASIC 轉機的時點明確，但 82.7 倍本益比沒有留下毛利率稀釋的空間"

FACTS = [
 "EPS TTM 60.69 元、年增 −9.2%；毛利率 46.2%、較四季前 −2.9pp（財報，2026Q2）",
 "**8 月營收年增 +44.1%**，為近五個月最陡加速（4 月 −4.1%／5 月 +5.0%／"
 "6 月 +2.8%／7 月 +12.2%／8 月 +44.1%）；月營收 TTM 年增仍僅 7.7%（低基期尚未消化）",
 "第一顆 AI 加速器 ASIC 2026Q4 量產（公司法說會）",
 "AI ASIC 2027 市佔目標由 10–15% 上修至 15–20%（公司法說會）",
 "2026 資料中心營收目標逾 20 億美元（公司法說會）",
 "與 Google TPUv8 合作，負責部分 I/O Die 設計與系統整合（公司）",
 "**本益比 82.74**（2026-09-21 收盤 5,010 元，創歷史新高）；"
 "論點初寫時為 74.8 —— 三個交易日內倍數擴張 10.6%，而 EPS 未動（市場價格）",
 "外資調升目標價至 10,000 元（媒體引述，非公司指引）",
]

INFERENCE = (
 "市場已先給足轉機評價，財報還沒轉：EPS TTM 年增 −9.2%、毛利率 46.2% 且年減 2.9pp。"
 "催化劑明確且落在窗口內 —— 第一顆 AI 加速器 ASIC 2026Q4 量產，"
 "2026 資料中心營收目標逾 20 億美元，ASIC 2027 市佔目標由 10–15% 上修到 15–20%。\n\n"
 "本次唯一的實質新證據是 8 月營收年增 +44.1%，這一串的前四個月分別是 "
 "−4.1%／+5.0%／+2.8%／+12.2% —— 加速是真的，轉機論點賭的那件事開始在月營收上出現。"
 "但 TTM 年增仍只有 7.7%，低基期要再幾個月才消化得完，所以這是「開始」不是「已經」。\n\n"
 "反方向同樣具體：本益比從論點寫下時的 74.8 升到 82.74，三個交易日擴張 10.6%，"
 "而分子一動也沒動。ASIC 的毛利率結構性低於手機 SoC —— 營收放大與毛利率稀釋"
 "會同時發生，82.7 倍留下的空間比 74.8 倍更少。兩股力量方向相反、量級相當，"
 "因此不改 P漲，只把前提改寫成它本來就該有的樣子。\n\n"
 "前提的寫法是這次的重點：原本兩條「賭它翻過來」的前提被寫成「賭它維持」，"
 "於是論點從下注第一天起就顯示失效。轉機型前提標成 expects=\"turn\"，"
 "未達成就是未達成，不是被推翻 —— 真正倒掉的是毛利率那條，那才是我該怕的事。"
)

CPS = [
 # 唯一的 hold 型前提，也是這個論點真正的風險所在：
 # 營收放大若以毛利率崩落為代價，論點就錯了，而且是錯在方向上。
 {"claim": "毛利率 ≥40%（ASIC 稀釋未失控）", "metric": "gross_margin",
  "op": ">=", "threshold": 0.40},
 # 以下兩條是轉機型：寫下當天本來就是假的，那正是論點的內容。
 {"claim": "EPS TTM 年增轉正（ASIC 確實接上）", "metric": "eps_yoy",
  "op": ">=", "threshold": 0.0, "expects": "turn"},
 {"claim": "月營收 TTM 年增 ≥15%（Q4 量產後低基期消化完）", "metric": "rev_yoy_ttm",
  "op": ">=", "threshold": 0.15, "expects": "turn"},
 # 新增的 hold 型：月營收年增不跌回個位數。8 月已是 +44.1%，
 # 門檻取 +10%（現值之下有充分緩衝），倒了才代表 8 月是單月假訊號。
 {"claim": "月營收年增 ≥10%（8 月的 +44.1% 不是單月跳動）", "metric": "rev_yoy",
  "op": ">=", "threshold": 0.10},
]


def main() -> None:
    import pandas as pd
    import checkpoints as CP

    src = ROOT / "judgments" / f"{AS_OF}_1y.json"
    d = json.loads(src.read_text(encoding="utf-8"))
    j = next(x for x in d["judgments"] if x["code"] == CODE)

    # 與 roll_1y 同一條式子：一年期只有一個原始量 P漲，其餘全由它與 σ 導出。
    pr = price_1y(P_UP, float(j["sigma_annual"]))
    ev, up, dn, q10, q90 = pr["median"], pr["up"], pr["dn"], pr["q10"], pr["q90"]

    j.update({
        "prob_up": round(P_UP, 4),
        "stance": "bullish" if ev >= 0 else "bearish",
        "thesis": THESIS, "facts": FACTS, "inference": INFERENCE,
        "rationale": INFERENCE,
        "falsifier": "；".join(c["claim"] for c in CPS)
                     + " —— hold 型任一被推翻即論點動搖；turn 型未達成僅代表轉機尚未發生",
        "researched": True, "conviction": "low",
        "checkpoints": CPS,
        "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
        "exp_ret": round(ev, 6), "mean_ret": round(pr["mean"], 6),
        "ret_q10": round(q10, 6), "ret_q90": round(q90, 6),
        "thesis_as_of": AS_OF, "repriced_only": False,
        "roll_note": "檢查點誤標為 hold 型導致第一天即失效，改標 expects=turn 並補上月營收 hold 條；P漲 不變",
    })

    # day-one 關卡：hold 型不得在基準日就被推翻。turn 型算「未達成」不擋。
    sc = CP.score_all(j, pd.Timestamp("2026-09-21"), "TW")
    bad = [c for c in sc["checkpoints"] if c["status"] == "broken"]
    if bad:
        print("！新檢查點在基準日就被推翻，請改門檻：")
        for c in bad:
            print(f"   {c['claim']}　實際 {c['actual']:+.3f}")
        sys.exit(1)

    d["judgments"] = sorted(d["judgments"], key=lambda x: -x["exp_ret"])
    src.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")

    # 狀態檔同步：滾動時 P漲 取自狀態檔，只改判斷檔等於沒改。
    st_p = ROOT / "data/roll_1y_state.json"
    st = json.loads(st_p.read_text(encoding="utf-8"))
    stocks = (st["stocks"] if "stocks" in st
              else st.get("markets", {}).get("TW", {}).get("stocks", {}))
    node = stocks.get(CODE)
    if node:
        node["prob_up"] = round(P_UP, 4)
        node["thesis_as_of"] = AS_OF
        node["broken"] = sc["broken"]
        st_p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"狀態檔已同步：{CODE} broken → {sc['broken']}")
    else:
        print(f"！狀態檔沒有 {CODE}")

    print(f"{CODE} 已修訂：{sc['verdict']}　holding {sc['holding']}／"
          f"未達成 {sc['unmet']}／broken {sc['broken']}　"
          f"P漲 維持 {P_UP}　期望值 {ev:+.2%}")


if __name__ == "__main__":
    main()
