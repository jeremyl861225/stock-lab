# -*- coding: utf-8 -*-
"""美股一年期（250 交易日）判斷 · 基準日 2026-09-18。

與台股那份（build_1y_20260916.py）同一套方法，但有三個市場差異必須先講清楚，
否則會拿台股的直覺去讀美股的數字：

1. **檢查點一年只能驗四次。**
   台股的月營收每月 10 日前公告，加上季報，一年約 16 次對帳機會。
   美國公司不公告月營收，全部繫於季報 —— 一年 4 次。
   所以這裡的檢查點刻意寫在「會在不同季度先後翻掉」的地方
   （毛利率、營益率、單季營收年增各管一件事），
   而不是三條都指向同一個營收數字。

2. **本益比會被一次性利益灌水，而且灌得很嚴重。**
   Alphabet 2026Q2 的淨利裡有約 990 億美元是權益證券的未實現評價利益，
   帳面本益比因此是 17.4 —— 用稅後營業利益重算是 36.9，差了一倍以上。
   凡是 features/us_fundamentals 標記 nonop_heavy 的公司，
   估值一律改用營業利益本益比，並在論點文字裡講明換過。

3. **可用日是法定上限，不是實際申報日。**
   10-Q 季末後 40 天、10-K 年度結束後 60 天。實際多半提早 2–3 週，
   所以我們會比市場晚知道前提翻掉。要拿真正的申報日得走 SEC EDGAR
   的 XBRL companyfacts，那需要在 User-Agent 放聯絡信箱，未取得同意前不做。

排除（照實說，不讓它們靜默消失）：
  · ETF（VOO／QQQ／BTCO，合計權重約 4.6%）—— 沒有公司財報可判斷。
    指數本身的一年期判斷是另一件事，不該用個股規則假裝做得出來。
  · 純金融（BRK-B／JPM／BAC／MS／GS／WFC／C／AXP，合計約 8.6%）——
    損益表沒有毛利率與營業利益兩列，而毛利率是本規則權重最高的訊號。
    V 與 MA 保留：它們是支付網路，有完整的毛利率與營益率。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from models.quantiles import quantiles
from models.rule_1y import ANCHOR, FLOOR, CAP, WEIGHTS
from features import us_fundamentals as UF

AS_OF, HORIZON, MARKET = "20260918", 250, "US"

EXCLUDE_ETF = {"VOO", "QQQ", "BTCO"}
EXCLUDE_BANK = {"BRK-B", "JPM", "BAC", "MS", "GS", "WFC", "C", "AXP"}

CP = lambda claim, metric, op, thr: {
    "claim": claim, "metric": metric, "op": op, "threshold": thr}

# ── 已實查前瞻計畫（來源逐條寫在 THESIS 的 facts）────────────────────
# code, p_up, skew, conviction, 推論, [檢查點]
J = [
 ("NVDA", 0.60, 0.08, "medium",
  "一年期窗口（2026Q4–2027Q3）落在「已被合約與產能鎖定的需求」與「2027 年高基期」"
  "的交界。多方是公司自己給的數字：Q3 財測 1,080 億美元、較去年同期增約九成，"
  "而且明講不含任何中國資料中心營收 —— 中國是選擇權不是風險。"
  "空方是算術：季營收要從 962 億再翻一倍，需要下游資本支出再增加約 3,000 億美元，"
  "那已經是賣方對 2027 年全年增量的全部。所以一年後最可能的狀態不是失速，"
  "是「仍在成長但增速腰斬」，屆時 28 倍本益比合不合理，全看毛利率守不守得住 74%。",
  [CP("毛利率 ≥70%（公司財測 74% 再留 4pp 緩衝）", "gross_margin", ">=", 0.70),
   CP("營益率 ≥60%（研發與代工成本未侵蝕）", "op_margin", ">=", 0.60),
   CP("單季營收年增 ≥40%（減速但未失速）", "rev_yoy", ">=", 0.40)]),

 ("AAPL", 0.50, -0.02, "medium",
  "記憶體漲價這一輪，蘋果站在成本那一側。服務佔比上升確實推升了毛利率"
  "（較四季前 +3.6pp），但一年期窗口裡有兩個反向力量同時作用："
  "公司自己預告供應限制將顯著增加，以及 DRAM／NAND 合約價在 2026 年大漲"
  "（見 MU 論點，那正是蘋果的成本）。用調漲售價轉嫁是可行的，但會壓抑銷量。"
  "38.5 倍本益比搭配公司自己給的 9–11% 營收成長，等於市場已假設"
  "「服務擴張」與「漲價不傷量」兩件事同時成立。不宣稱方向。",
  [CP("毛利率 ≥47%（記憶體成本未吃掉服務帶來的改善）", "gross_margin", ">=", 0.47),
   CP("單季營收年增 ≥5%（落在公司財測下緣之下仍可接受）", "rev_yoy", ">=", 0.05),
   CP("EPS 年增維持正值", "eps_yoy", ">=", 0.0)]),

 ("GOOGL", 0.55, 0.03, "medium",
  "本批最容易被數字騙到的一檔。2026Q2 帳面淨利年增 298%，其中約 990 億美元是"
  "權益證券的**未實現**評價利益 —— 它讓任何以淨利為分母的估值指標失真一倍以上，"
  "而且明年不會重複，甚至可能反向。剝掉之後，Alphabet 是一家營業利益成長 30%、"
  "雲端成長 82%、資本支出已佔營收近三成的公司，稅後營業利益本益比 36.9 而非 17.4。"
  "一年期的問題不是「AI 有沒有用」，是折舊開始認列時營益率守不守得住 —— "
  "資本支出的損益表代價落後 12–18 個月，正好落在這個窗口。",
  [CP("營益率 ≥30%（折舊開始認列後仍守住）", "op_margin", ">=", 0.30),
   CP("單季營收年增 ≥15%", "rev_yoy", ">=", 0.15),
   CP("毛利率 ≥58%", "gross_margin", ">=", 0.58)]),

 ("MSFT", 0.53, 0.02, "medium",
  "毛利率已經在降了 —— 較四季前 −1.4pp，是本批權重前十裡唯一毛利率轉負的。"
  "這不是意外，是把約 1,900 億美元的資本支出變成可出租算力的必然代價："
  "折舊與電力先進成本，營收要等客戶把容量用滿才追上。"
  "27.7 倍本益比在超大規模業者裡不算貴，但它假設這段落差會收斂，"
  "而落差收斂的時點取決於企業端 AI 用量，那是目前最沒有硬數字的一件事。",
  [CP("毛利率 ≥65%（折舊壓縮未超過 2pp）", "gross_margin", ">=", 0.65),
   CP("營益率 ≥42%", "op_margin", ">=", 0.42),
   CP("單季營收年增 ≥12%", "rev_yoy", ">=", 0.12)]),

 ("AMZN", 0.55, 0.04, "medium",
  "四家超大規模業者裡資本支出最高（2026 年約 2,000 億美元），但營益率 13.1% "
  "也是最低 —— 零售本業把整體比率壓下來，AWS 的獲利被稀釋在合併報表裡。"
  "本益比 20.4 是四家裡最低的一個，反映的正是這件事。"
  "一年期看的是 AWS 成長能不能快到讓合併營益率往上走。"
  "**注意**：上游資料的 2026Q2 一列只有 EPS、其餘全缺，本判斷因此建立在"
  "已申報的 2026Q1 上，比其他標的舊一季。",
  [CP("營益率 ≥10%（資本支出未吃掉零售的獲利）", "op_margin", ">=", 0.10),
   CP("單季營收年增 ≥10%", "rev_yoy", ">=", 0.10),
   CP("毛利率 ≥48%", "gross_margin", ">=", 0.48)]),

 ("AVGO", 0.58, 0.06, "medium",
  "公司把 2027、2028 的數字講死了 —— AI 營收 FY27 1,150 億、FY28 2,300 億美元，"
  "依據是「已確保的供給與 XPU 部署能見度」，在半導體業罕見。"
  "但 1,150 億對 FY26 的 580 億意味著要再翻倍，而其中相當比例繫於少數幾個客戶的"
  "自研加速器計畫。客戶集中度是這個論點的單一最大風險，"
  "而它不會先反映在毛利率上 —— 它會在某個客戶改變計畫時一次出現。"
  "45.6 倍本益比對應的是「兩年翻四倍」的路徑被接受。",
  [CP("毛利率 ≥65%（客製 ASIC 佔比上升的稀釋在可控範圍）", "gross_margin", ">=", 0.65),
   CP("單季營收年增 ≥50%（FY27 翻倍路徑未落後）", "rev_yoy", ">=", 0.50),
   CP("營益率 ≥48%", "op_margin", ">=", 0.48)]),

 ("MU", 0.48, -0.04, "medium",
  "記憶體的歷史教訓是「本益比最低的時候最貴」——分母是週期高點的獲利。"
  "84.6% 的毛利率比 2017–18 那次超級循環的高點還高出約 20 個百分點，"
  "而公司自己揭露的合約毛利率下限是 60%：兩者之間有 24 個百分點的落差，"
  "那個落差就是一年後最可能發生的事。合約 backlog 逾 1,000 億美元、"
  "HBM3E 售罄至 2027Q4，確實把未來五個季度的**營收**鎖住了，所以我不看空營收；"
  "我看空的是「毛利率維持在 84.6%」這個被 22.9 倍（前瞻約 9 倍）本益比隱含的假設。"
  "P漲 取 0.48 不是預測下跌，是說在這個位置上，上檔與下檔不對稱地偏向下檔。",
  [CP("毛利率 ≥60%（公司自己揭露的合約毛利率下限）", "gross_margin", ">=", 0.60),
   CP("營益率 ≥50%", "op_margin", ">=", 0.50),
   CP("單季營收年增 ≥50%（合約如期出貨）", "rev_yoy", ">=", 0.50)]),

 ("META", 0.50, 0.00, "medium",
  "本批裡「花錢已經開始影響損益、收入還沒出現」最明確的一檔："
  "2026 年資本支出指引年內二度上調至 1,350–1,450 億美元，"
  "同期 EPS 年增 −13.4%、毛利率較四季前 −0.8pp、營益率 30.9%"
  "（在毛利率 81.4% 的公司身上，這個營益率偏低）。"
  "25.7 倍本益比不貴，但它對應的是一家獲利正在下滑的公司 —— "
  "便宜與變便宜是兩件事。不宣稱方向。",
  [CP("營益率 ≥28%", "op_margin", ">=", 0.28),
   CP("毛利率 ≥78%", "gross_margin", ">=", 0.78),
   CP("EPS 年增 ≥−20%（下滑未再擴大一倍）", "eps_yoy", ">=", -0.20)]),

 ("TSM", 0.60, 0.08, "medium",
  "與台股判斷裡的 2330 是**同一家公司**，論點相同：營收成長蓋不蓋得過"
  "管理層已預告的毛利率壓縮（N2 量產稀釋 3–4pp、海外廠再稀釋 2–3pp）。"
  "2026Q2 毛利率 67.7%、營益率 60.3%、單季營收年增 36.0%，資本支出強度 34.0%。"
  "**這一檔與 2330 不是兩個賭注。**同時持有台股 2330 與美股 TSM 不是分散，"
  "是把同一個判斷押兩次；兩邊的權重要合起來看（0050 裡 51.5% ＋ 本清單 4.7%）。",
  [CP("毛利率 ≥60%（稀釋未超出管理層預告的 7pp）", "gross_margin", ">=", 0.60),
   CP("營益率 ≥55%", "op_margin", ">=", 0.55),
   CP("單季營收年增 ≥25%（AI 需求未轉弱）", "rev_yoy", ">=", 0.25)]),
]

THESIS = {
 "NVDA": ("成長必然減速，一年後的估值全押在毛利率守不守得住 74%", [
   "2026Q2（截至 7/26）營收 962.2 億美元、資料中心 890 億美元、年增 117%（公司財報）",
   "Q3 財測營收 1,080 億美元 ±2%、非 GAAP 毛利率 74.0% ±50bp（公司財測）",
   "財測明載不假設任何來自中國的資料中心運算營收（公司）",
   "資料中心結構：超大規模 487 億、AI 雲與企業 403 億美元（公司）",
   "四大雲端 2026 年資本支出合計約 7,540 億美元、年增 83%；2027 年賣方共識約 9,050 億美元（賣方共識，非公司指引）",
   "毛利率 75.0%、營益率 66.2%、本益比 28.1（財報與市場價格）"]),
 "AAPL": ("服務擴張與記憶體漲價方向相反，而 38.5 倍已把兩件事都算成功", [
   "FY26 Q4 財測營收年增 9–11%，低於市場原估 12%（公司財測）",
   "公司預告供應限制將「顯著增加」，影響 iPhone、Mac、iPad（公司法說會）",
   "FY26 Q3 服務營收 307.4 億美元創新高；服務毛利率 75.3% vs 產品 36.2%（公司財報）",
   "賣方推估 iPhone 18 Pro 售價將調漲 100–200 美元，主因記憶體成本（賣方推估，非公司指引）",
   "毛利率 50.1%、較四季前 +3.6pp；單季營收年增 16.4%；本益比 38.5"]),
 "GOOGL": ("帳面 17 倍是一次性評價利益造成的假象，實際 37 倍，而資本支出已開始被收費", [
   "2026Q2 淨利 1,121 億美元、年增 298%，其中約 990 億是權益證券**未實現**評價利益（公司財報）",
   "同期營業利益 407.7 億美元、年增 30%；營收 1,197.9 億美元、年增 24%（公司財報）",
   "Google Cloud 營收 248 億美元、年增 82%（公司財報）",
   "2026 年資本支出指引上調至約 2,050 億美元；財報當日股價因資本支出上調而下跌（公司指引／市場反應）",
   "帳面本益比 17.4，以稅後營業利益計為 36.9；資本支出強度 29.7%"]),
 "MSFT": ("毛利率已經開始被折舊壓縮，而估值假設這段落差會在一年內收斂", [
   "2026 年資本支出約 1,900 億美元（市場彙整之公司指引）",
   "毛利率 67.2%、較四季前 −1.4pp；營益率 45.1%；單季營收年增 17.7%（財報）",
   "本益比 27.7（市場價格）"]),
 "AMZN": ("資本支出最高、營益率最低，一年期看 AWS 能不能把合併營益率拉起來", [
   "2026 年資本支出約 2,000 億美元（市場彙整之公司指引）",
   "營益率 13.1%、毛利率 51.8%、單季營收年增 16.6%（2026Q1 財報；Q2 上游資料不全）",
   "本益比 20.4，為四大超大規模業者最低（市場價格）"]),
 "AVGO": ("公司把 2027／2028 講死了，風險不在毛利率而在客戶集中度", [
   "FY26 AI 營收指引 580 億美元（自 560 億上調）、年增 186%（公司法說會）",
   "FY27 AI 營收展望 1,150 億、FY28 2,300 億美元，依據為已確保的供給與 XPU 部署能見度（公司法說會）",
   "AI 訂單 backlog 逾 730 億美元（公司）",
   "Q3 客製 AI 晶片營收 167 億美元、年增 221%；Q4 財測被市場視為不如預期（公司財報／市場反應）",
   "客戶含 Google（Ironwood TPU）、Meta（MTIA）、OpenAI、Anthropic、ByteDance、Fujitsu（公司法說會）",
   "毛利率 69.1%、營益率 54.3%、單季營收年增 85.5%、本益比 45.6"]),
 "MU": ("營收已被合約鎖住，但 84.6% 的毛利率沒有被鎖住，而估值假設它會留著", [
   "合約 backlog 逾 1,000 億美元、延伸至 2028 年，公司揭露之合約毛利率逾 60%（公司）",
   "HBM3E 產能售罄至 2027Q4（公司）",
   "2026Q3（截至 5/31）毛利率 84.6%、營益率 80.4%、單季營收年增 345.7%（財報）",
   "DRAM 合約價 2025→2027 預估上漲約 275–300%（賣方推估，非公司指引）",
   "本益比 22.9（TTM）；賣方對 FY27 的 EPS 共識約 112 美元，對應約 9 倍（賣方共識）"]),
 "META": ("資本支出年內二度上調，而 EPS 年增已經是負的", [
   "2026 年資本支出指引 1,350–1,450 億美元，年內二度上調（公司指引）",
   "EPS 年增 −13.4%；毛利率 81.4%、較四季前 −0.8pp；營益率 30.9%（財報）",
   "本益比 25.7（市場價格）"]),
 "TSM": ("與台股 2330 同一個判斷：成長蓋不蓋得過管理層已預告的毛利率壓縮", [
   "2026Q2 毛利率 67.7%、營益率 60.3%、單季營收年增 36.0%、資本支出強度 34.0%（財報）",
   "管理層預告 N2 量產稀釋毛利率 3–4pp、海外廠再稀釋 2–3pp（公司法說會）",
   "2026 資本支出由 520–560 上調至 600–640 億美元（公司法說會）",
   "本檔與台股 2330 為同一發行人，權重須合併計算（事實陳述）"]),
}


def _rule_based(d: pd.DataFrame) -> list:
    """未實查前瞻計畫的標的：用明確規則從財報推導，並在論點裡講明。

    與台股共用 models/rule_1y 的權重與錨點，但估值欄位在這裡先做過一次替換：
    被標記 nonop_heavy 的公司改用稅後營業利益本益比（見檔頭第 2 點）。
    不替換的話，Alphabet 這類公司會因為一次性評價利益而被排進「最便宜」的一端。
    """
    out = []

    def pct(col, invert=False):
        if col not in d.columns or d[col].notna().sum() < 5:
            return pd.Series(0.0, index=d.index)
        r = d[col].rank(pct=True) - 0.5
        return (-r if invert else r).fillna(0.0)

    adj = sum(pct(c, invert=(c == "PER")) * w for c, w in WEIGHTS.items())

    for i, r in d.iterrows():
        if pd.isna(r.gm_chg_4q) and pd.isna(r.eps_yoy):
            continue                                   # 財報不足，不做判斷
        p = float(np.clip(ANCHOR + adj.loc[i], FLOOR, CAP))
        why = []
        for c, lbl, unit in (("gm_chg_4q", "毛利率較四季前", "pp"),
                             ("roe_ttm", "ROE", "%"),
                             ("eps_yoy", "EPS 年增", "%"),
                             ("rev_cagr_3y", "三年營收 CAGR", "%")):
            v = r[c]
            if pd.isna(v):
                continue
            q = int(d[c].rank(pct=True).loc[i] * 100)
            txt = f"{v*100:+.1f}pp" if unit == "pp" else f"{v*100:+.0f}%"
            if c == "eps_yoy":
                txt += f"（{r.eps_yoy_basis}）"
            why.append(f"{lbl}{txt}（同業第 {q} 百分位）")
        if not pd.isna(r.PER) and r.PER > 0:
            q = int((1 - d["PER"].rank(pct=True).loc[i]) * 100)
            tag = "營業利益本益比" if r.nonop_heavy else "本益比"
            why.append(f"{tag} {r.PER:.1f}（便宜度第 {q} 百分位）")
        note = ("；本益比已改用稅後營業利益計算，因帳面淨利含重大非營業損益"
                if r.nonop_heavy else "")

        cps = []
        if not pd.isna(r.gross_margin):
            cps.append(CP(f"毛利率 ≥{(r.gross_margin-0.03)*100:.1f}%（獲利結構未惡化）",
                          "gross_margin", ">=", round(float(r.gross_margin - 0.03), 4)))
        if not pd.isna(r.op_margin):
            cps.append(CP(f"營益率 ≥{(r.op_margin-0.03)*100:.1f}%（費用未失控）",
                          "op_margin", ">=", round(float(r.op_margin - 0.03), 4)))
        # 營收檢查點的門檻必須相對現值，不能一律寫「維持正值」。
        # 成長率沒有上下界，統一寫 0 會同時犯兩種錯：
        # 對年增 +85% 的公司太鬆（要掉 85 個百分點才算前提倒），
        # 對年增 −4% 的公司則是**出生當天就被推翻**（實測 VZ −0.7%、QCOM −4.0%）。
        # 已經被推翻的檢查點不是前提，是把一個現成的事實寫成了預測。
        if not pd.isna(r.rev_yoy):
            v = float(r.rev_yoy)
            if v > 0.20:        # 高成長：成長腰斬才算前提倒
                thr, txt = round(v / 2, 4), f"單季營收年增 ≥{v/2*100:.0f}%（成長未腰斬）"
            elif v > 0:         # 低成長：轉負就算前提倒
                thr, txt = 0.0, "單季營收年增維持正值"
            else:               # 已經衰退：再惡化 5pp 才算前提倒
                thr, txt = round(v - 0.05, 4), f"單季營收年增 ≥{(v-0.05)*100:.0f}%（衰退未擴大）"
            cps.append(CP(txt, "rev_yoy", ">=", thr))
        out.append((r.code, round(p, 3), round((p - ANCHOR) * 1.2, 3), "low",
                    "RULE:未實查公司前瞻計畫，本判斷由財報規則推導（橫斷面百分位）："
                    + "、".join(why) + note
                    + "。證據強度低於已實查標的，信心一律標低。",
                    cps))
    return out


def build() -> dict:
    p = pd.read_parquet(ROOT / "data/features/panel.parquet")
    p = p[p.market == MARKET].sort_values(["code", "date"])
    p["r"] = p.groupby("code")["close"].pct_change()
    vol60 = p.groupby("code")["r"].apply(lambda s: s.tail(60).std())

    b = pd.read_parquet(ROOT / "data/briefing.parquet")
    b = b[b.market == MARKET][["code", "名稱", "close", "PER"]]
    f = UF.as_of(pd.Timestamp(f"{AS_OF[:4]}-{AS_OF[4:6]}-{AS_OF[6:]}")).merge(
        b, on="code", how="inner")

    f = f[~f.code.isin(EXCLUDE_ETF | EXCLUDE_BANK)]
    # 估值替換：一次性利益灌水的公司改用稅後營業利益本益比。
    per_op = f["close"] / f["eps_op_ttm"]
    f["PER"] = np.where(f["nonop_heavy"] & per_op.gt(0) & per_op.notna(),
                        per_op, f["PER"])

    done = {j[0] for j in J}
    rows = J + [x for x in _rule_based(f) if x[0] not in done]

    out = []
    for code, p_up, skew, conf, why, cps in rows:
        v = vol60.get(code)
        if v is None or pd.isna(v) or v <= 0:
            continue
        base = 0.85 * float(v) * math.sqrt(HORIZON)
        up, dn = base * (1 + skew), -base * (1 - skew)
        ev = p_up * up + (1 - p_up) * dn
        sig = float(v) * math.sqrt(HORIZON)
        q10, q90 = quantiles(ev, sig)
        researched = not why.startswith("RULE:")
        why_clean = why.removeprefix("RULE:")
        th, facts = THESIS.get(code, (None, None))
        if th is None:
            th = "未實查前瞻計畫；僅依財報橫斷面位置推導"
            facts = [why_clean]
        falsifier = "；".join(c["claim"] for c in cps) + " —— 任一前提被推翻即論點動搖"
        out.append({
            "code": code, "market": MARKET, "prob_up": round(p_up, 4),
            "stance": "bullish" if ev >= 0 else "bearish",
            "thesis": th, "facts": facts, "inference": why_clean,
            "falsifier": falsifier, "researched": researched,
            "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
            "exp_ret": round(ev, 6),
            "ret_q10": round(q10, 6), "ret_q90": round(q90, 6),
            "conviction": conf, "rationale": why_clean, "checkpoints": cps,
        })
    out.sort(key=lambda x: -x["exp_ret"])
    return {
        "as_of": AS_OF, "horizon": HORIZON, "market": MARKET,
        "version": "1y-1.0.0",
        "market_context": (
            "這份清單看起來像 50 個獨立標的，其實主要是**一筆交易的兩側**。"
            "買方：MSFT、GOOGL、AMZN、META 四家 2026 年資本支出合計約 7,540 億美元、"
            "年增 83%，合計權重 25.6%。賣方：NVDA、TSM、AVGO、MU、AMD、ANET 合計"
            "權重約 24.1%，營收直接來自那筆錢。兩邊加起來接近本清單的一半。"
            "所以「分散在十檔 AI 股」不是分散 —— 若 2027 年的資本支出共識"
            "（賣方約 9,050 億美元）被下修，兩側會同時受傷，只是先後不同："
            "賣方先看到訂單，買方先看到股價。\n\n"
            "利率是第二個共同因子，而且方向與去年相反。聯準會自 2025 年 12 月降息後"
            "維持在 3.50–3.75%，經 8 月 Jackson Hole 的鷹派談話後，"
            "市場對「下一步是升息」的定價已超過五成。本清單裡本益比逾 40 倍的"
            "有 AVGO、COST、PANW、PLTR、TSLA、ABBV、MRK —— 倍數壓縮的風險"
            "不需要基本面變壞就會發生。\n\n"
            "TSM 與台股判斷裡的 2330 是同一家公司。同時持有不是分散，"
            "是把同一個判斷押兩次，權重要合起來看。\n\n"
            "**已知缺口，照實列出：**\n"
            "一、檢查點一年只能驗四次。美國公司不公告月營收，前提對帳全部繫於季報，"
            "而台股有月營收可每月對帳（約 16 次）。同一套機制在美股的回饋速度慢四倍。\n"
            "二、純金融（BRK-B／JPM／BAC／MS／GS／WFC／C／AXP，約佔本清單權重 8.6%）"
            "不在判斷內。它們的損益表沒有毛利率與營業利益，而毛利率是本規則權重最高的"
            "訊號；要另建 NIM／信用成本／淨值成長那套特徵才算數。\n"
            "三、ETF（VOO／QQQ／BTCO，約 4.6%）不在判斷內 —— 沒有公司財報可判斷，"
            "指數的一年期判斷是另一件事，不該用個股規則假裝做得出來。\n"
            "四、可用日用法定申報上限（10-Q 40 天、10-K 60 天），實際申報多半提早 2–3 週，"
            "所以我們會比市場晚知道前提翻掉。\n"
            "五、Alphabet 2026Q2 帳面淨利含約 990 億美元未實現評價利益，"
            "本批已把它的估值改用稅後營業利益計算（17.4 倍 → 36.9 倍）；"
            "同樣處理的還有 MRK 與 TSLA。"),
        "judgments": out,
    }


def _audit_day_one(d: dict) -> list:
    """檢查點在**寫下當天**就被推翻，是建構錯誤，不是發現。

    前提的意義是「接下來若這件事翻掉，代表我錯了」。若它在下注當天
    就已經是假的，那它不是前提，是我把一個現成的事實寫成了預測 ——
    而且它會讓面板一開張就顯示「論點動搖」，把真正的推翻訊號淹掉。
    （已實查論點的前提被**新資料**推翻是另一回事，那是機制在運作。）
    """
    import checkpoints as CP_
    today = pd.Timestamp(f"{AS_OF[:4]}-{AS_OF[4:6]}-{AS_OF[6:]}")
    bad = []
    for j in d["judgments"]:
        for c in CP_.score_all(j, today, MARKET)["checkpoints"]:
            if c["status"] == "broken":
                bad.append((j["code"], c["claim"], c["actual"]))
    return bad


if __name__ == "__main__":
    d = build()
    bad = _audit_day_one(d)
    if bad:
        print("！以下檢查點在基準日就已被推翻，屬建構錯誤，請先修門檻：")
        for code, claim, act in bad:
            print(f"   {code:<6}{claim}　實際 {act:+.3f}")
        sys.exit(1)
    out = Path(__file__).resolve().parent / f"{AS_OF}_us_1y.json"
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    n_res = sum(j["researched"] for j in d["judgments"])
    n_cp = sum(len(j["checkpoints"]) for j in d["judgments"])
    print(f"美股一年期判斷：{len(d['judgments'])} 檔 → {out.name}")
    print(f"  檢查點 {n_cp} 條；已實查 {n_res} 檔、規則推導 {len(d['judgments'])-n_res} 檔")
    print(f"\n  {'代號':<7}{'P漲':>6}{'漲幅':>9}{'跌幅':>9}{'期望值':>9}  信心")
    for j in d["judgments"]:
        mark = "＊" if j["researched"] else "　"
        print(f"  {j['code']:<7}{j['prob_up']*100:>5.0f}%{j['up_magnitude']*100:>+8.1f}%"
              f"{j['dn_magnitude']*100:>+8.1f}%{j['exp_ret']*100:>+8.2f}%  {j['conviction']}{mark}")
