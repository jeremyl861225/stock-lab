"""一年期（250 交易日）判斷 · 基準日 2026-09-16。

與 5／20 日判斷的差別，不只是期別拉長：

  5／20 日：技術面、籌碼面、動能。問「這批資金會不會繼續進來」。
  250 日  ：基本面、公司自己的擴張計畫、產業循環位置。
            問「這家公司一年後會不會比現在賺得多，而現在的價格有沒有先反映」。

這裡只寫判斷，不套統計模型 —— 兩年面板在 250 日尺度上每檔只有約 2 筆
獨立觀測，擬合出來的係數是雜訊（見 config.py 的 HORIZON_1Y 註解）。

**每一檔都必須寫檢查點。** 一年期預測要到 2027 年才結算，若只押方向然後
等，這一年裡完全沒有回饋。檢查點把論點拆成每月／每季可自動對帳的前提：
前提被推翻就當場知道論點壞了，不必等到期。說不出前提的看多，
通常不是判斷而是氛圍。

幅度：base = 0.85 × vol_60 × √250
  · vol_60 而非 vol_20 —— 實測預測「未來 120 日實現波動」的能力
    vol_60 r=0.908 > vol_250 r=0.883 > vol_120 r=0.869 > vol_20 r=0.841。
    短期波動會均值回歸，拿它外推一年會高估 7.1 個百分點。
  · √250 外推站得住 —— 實測變異比 VR(h) 在 h=5~120 都落在 0.96–1.00，
    報酬接近隨機漫步，無顯著均值回歸。

基準機率：中性錨點取 0.55 而非 0.50。
  樣本內 250 日為正的比例是 79%，但那是兩年單一多頭，嚴重高估、不可用。
  改用先驗：大盤 12 個月為正約 65–70%，個股低於大盤
  （Bessembinder 2018 JFE：多數個股長期報酬中位數低於指數），故取 0.55。

信心分級一律以「有沒有實查該公司的前瞻計畫」為準：
  medium = 已查管理層公開財測／擴產計畫（見 research/1y_notes.md）
  low    = 只有財報軌跡與估值，未查前瞻計畫
把沒查過的標的寫得跟查過的一樣有把握，是這套系統最該避免的事。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.quantiles import quantiles

AS_OF, HORIZON = "20260916", 250
ANCHOR = 0.55

# code, p_up, skew, conviction, thesis, [checkpoints]
CP = lambda claim, metric, op, thr: {
    "claim": claim, "metric": metric, "op": op, "threshold": thr}

J = [
 # ══ 已實查前瞻計畫（research/1y_notes.md）══════════════════════════
 ("2330", 0.60, 0.08, "medium",
  "營收與毛利率在未來一年方向相反。多方是公司自己押的錢：資本支出從 520–560 "
  "上調到 600–640 億美元，CoWoS 2027 年前擴產逾 60%，N2 產能規劃比技術論壇公布的更大。"
  "空方是管理層自己預告的：N2 量產稀釋毛利率 3–4pp、海外廠再稀釋 2–3pp，"
  "67.7% 扣掉即 61–63%；CoWoS 供需缺口從 20% 收斂到年底 10%，短缺緩解會鬆動定價權。"
  "PER 27.6 搭配 EPS +53% 並不昂貴，故偏多，但關鍵在成長蓋不蓋得過壓縮。",
  [CP("月營收 TTM 年增維持 ≥25%（AI 需求未轉弱）", "rev_yoy_ttm", ">=", 0.25),
   CP("毛利率 ≥60%（稀釋未超出管理層預告的 7pp）", "gross_margin", ">=", 0.60),
   CP("營益率 ≥55%（費用未隨海外擴張失控）", "op_margin", ">=", 0.55),
   # 多方論點的核心是「公司自己拿真錢押」。資本支出縮手＝論點反轉，
   # 而且這會在財報上比營收更早出現。目前 33.6%。
   CP("資本支出強度 ≥28%（擴張未收手）", "capex_intensity", ">=", 0.28)]),

 ("2454", 0.52, -0.02, "low",
  "市場已先給轉機評價（PER 74.8），財報還沒轉（EPS TTM 年增 −9.2%、毛利率 46.2% 且年減 2.9pp）。"
  "催化劑明確且落在窗口內：第一顆 AI 加速器 ASIC 2026Q4 量產，2026 資料中心營收目標逾 20 億美元，"
  "ASIC 2027 市佔目標由 10–15% 上修到 15–20%。但 ASIC 毛利率結構性低於手機 SoC —— "
  "營收放大與毛利率稀釋會同時發生，而 74.8 倍的本益比沒有留下稀釋的空間。",
  [CP("EPS TTM 年增轉正（ASIC 確實接上）", "eps_yoy", ">=", 0.0),
   CP("毛利率 ≥40%（稀釋未失控）", "gross_margin", ">=", 0.40),
   CP("月營收 TTM 年增 ≥15%（量產未遞延）", "rev_yoy_ttm", ">=", 0.15)]),

 ("2308", 0.57, 0.06, "medium",
  "三檔權值股裡催化劑時點最明確的一檔。800V DC 普遍預期 2027 年大量普及，"
  "正好落在一年期窗口（2026Q4–2027Q3）的起點；法人估 2027 年 AI 營收佔比上看五成。"
  "公司押的注對得上同一時點：2026 資本支出約 700 億元，較 2025 年的 461 億增加 52%。"
  "AI 產品佔營收已逾 25%、液冷由 10% 升至逾 12%。風險是 PER 54.4 已反映一部分，"
  "且 800VDC 若遞延到 2028 會先反映在營收成長上。",
  [CP("毛利率 ≥35%（AI 電源與液冷佔比上升應撐住）", "gross_margin", ">=", 0.35),
   CP("月營收 TTM 年增 ≥20%（800VDC 未遞延）", "rev_yoy_ttm", ">=", 0.20),
   CP("營益率 ≥14%（擴產費用未吃掉獲利）", "op_margin", ">=", 0.14),
   # 2026 資本支出 700 億 vs 2025 的 461 億是論點核心，目前強度 8.2%。
   CP("資本支出強度 ≥7%（700 億擴產計畫未縮手）", "capex_intensity", ">=", 0.07)]),

 ("2317", 0.58, 0.06, "medium",
  "AI 伺服器鏈裡估值最低的一檔（PER 16.4）。產業層級的關鍵事實是：主要 ODM 廠"
  "2026H2 及 2027 年新產能幾乎已被預訂完畢 —— 成長上限是產能而非需求，"
  "變數因此從『需求會不會來』移到『擴產能不能如期』。鴻海在美國、墨西哥、台灣"
  "建置 L6–L10／L11 產能。月營收年增 +52%、EPS +16%。"
  "毛利率 6.1% 本就極薄，任何組裝代工的價格壓力都會直接穿透到獲利。",
  [CP("月營收 TTM 年增 ≥25%（產能如期開出）", "rev_yoy_ttm", ">=", 0.25),
   CP("毛利率 ≥5.5%（代工價格壓力未穿透）", "gross_margin", ">=", 0.055),
   CP("EPS TTM 年增維持正值", "eps_yoy", ">=", 0.0)]),

 ("2382", 0.57, 0.05, "medium",
  "訂單能見度直達 2027，加州與田納西新廠年底前三座投運，AI 伺服器產能較去年翻倍，"
  "月營收年增 +177.5%，PER 僅 14.8。最該盯的是毛利率：目前 5.0% 且年減 2.0pp —— "
  "營收翻倍而毛利率下滑，代表這一輪擴張是用價格換量。若毛利率續跌，"
  "營收成長再高也不會轉成 EPS。",
  [CP("毛利率 ≥4.5%（擴張不是純粹殺價換量）", "gross_margin", ">=", 0.045),
   CP("月營收 TTM 年增 ≥40%（產能翻倍反映在營收）", "rev_yoy_ttm", ">=", 0.40),
   CP("EPS TTM 年增 ≥15%（量增有轉成獲利）", "eps_yoy", ">=", 0.15)]),

 ("3231", 0.55, 0.03, "low",
  "與廣達同一條鏈、同一個共同因子（北美雲端資本支出），偕緯穎在北美與東南亞建 L10／L11，"
  "竹北高雄設研發測試中心。月營收年增高，但未實查其個別財測與產能時程，"
  "論點強度低於廣達，故信心標低。",
  [CP("月營收 TTM 年增 ≥30%", "rev_yoy_ttm", ">=", 0.30),
   CP("毛利率不再下滑（較四季前持平或改善）", "gm_chg_4q", ">=", -0.005)]),

 ("3037", 0.53, 0.02, "low",
  "ABF 載板訂單能見度延伸至 2027，毛利率 24.8% 且較四季前大增 11.7pp，"
  "EPS 由極低基期暴增。但 PER 63.98 已把復甦反映得相當充分，"
  "且載板是典型的擴產競賽產業 —— 能見度好的時候大家一起擴，兩年後供給就回來了。",
  [CP("毛利率 ≥22%（載板報價未反轉）", "gross_margin", ">=", 0.22),
   CP("月營收 TTM 年增 ≥25%", "rev_yoy_ttm", ">=", 0.25)]),

 ("3189", 0.50, 0.00, "low",
  "同為 ABF 載板、訂單能見度至 2027，但 PER 152 —— 即使復甦如期，"
  "這個評價也沒有留下失誤的空間。不宣稱方向。",
  [CP("毛利率 ≥24%", "gross_margin", ">=", 0.24),
   CP("月營收 TTM 年增 ≥25%", "rev_yoy_ttm", ">=", 0.25)]),
]


# 已實查標的的 thesis 與 facts。facts 一律寫「誰說的、數字多少」——
# 一年期判斷最容易出的錯，是把分析師預估或自己的推論寫成事實。
THESIS = {
 "2330": ("營收成長蓋得過管理層已預告的毛利率壓縮",
   ["2026Q2 毛利率 67.7%、EPS 27.25 元，皆為歷史新高（公司法說會）",
    "2026 資本支出由 520–560 上調至 600–640 億美元（公司法說會）",
    "CoWoS 產能 2027 年前擴產逾 60%，2022–2027 CAGR >80%（公司）",
    "管理層預告：N2 量產稀釋毛利率 3–4pp、海外廠再稀釋 2–3pp（公司法說會）",
    "CoWoS 供需缺口由 20% 收斂至 2026 年底 10%（TrendForce）",
    "月營收 TTM 年增 +32.2%、PER 27.6"]),
 "2454": ("ASIC 轉機的時點明確，但 74.8 倍本益比沒有留下毛利率稀釋的空間",
   ["EPS TTM 年增 −9.2%、毛利率 46.2% 且較四季前 −2.9pp（財報）",
    "第一顆 AI 加速器 ASIC 2026Q4 量產（公司法說會）",
    "AI ASIC 2027 市佔目標由 10–15% 上修至 15–20%（公司法說會）",
    "2026 資料中心營收目標逾 20 億美元（公司法說會）",
    "與 Google TPUv8 合作，負責部分 I/O Die 設計與系統整合（公司）"]),
 "2308": ("800VDC 的放量起點正好落在一年期窗口，且公司的資本支出對得上同一時點",
   ["AI 產品佔營收已逾 25%、液冷由 2025 年約 10% 升至逾 12%（公司法說會）",
    "2026 資本支出約 700 億元，較 2025 年 461 億增加 52%（公司）",
    "800V DC 普遍預期 2027 年大量普及（產業共識，非公司指引）",
    "毛利率 35.6%、EPS TTM 年增 +86.8%、PER 54.4（財報）"]),
 "2317": ("估值最低的 AI 伺服器代工，而該鏈 2027 年產能已被訂滿",
   ["主要 ODM 廠 2026H2 及 2027 新產能幾乎已被預訂完畢（產業報導）",
    "在美國、墨西哥、台灣建置 L6–L10／L11 產能（公司）",
    "毛利率 6.1%、EPS TTM 年增 +16.0%、月營收年增 +52%、PER 16.4（財報）"]),
 "2382": ("產能翻倍且能見度到 2027，但毛利率下滑顯示這一輪是用價格換量",
   ["訂單能見度直達 2027、AI 伺服器產能較去年翻倍（公司）",
    "加州、田納西新廠年底前三座投運（公司）",
    "毛利率 5.0% 且較四季前 −2.0pp、月營收年增 +177.5%、PER 14.8（財報）"]),
 "3231": ("與廣達同鏈同因子，但未實查其個別財測與產能時程",
   ["偕緯穎在北美、東南亞建 L10／L11，竹北高雄設研發測試中心（產業報導）",
    "未取得公司自身的產能時程與財測"]),
 "3037": ("ABF 復甦已被 64 倍本益比充分反映，且載板是典型的擴產競賽產業",
   ["ABF 載板訂單能見度延伸至 2027（產業報導）",
    "毛利率 24.8%、較四季前 +11.7pp，EPS 由極低基期回升（財報）",
    "PER 63.98（市場價格）"]),
 "3189": ("即使復甦如期，152 倍本益比也沒有留下失誤的空間",
   ["ABF 載板訂單能見度至 2027（產業報導）",
    "PER 152.13（市場價格）"]),
}


def _rule_based(f: pd.DataFrame) -> list:
    """未實查前瞻計畫的標的：用明確規則從財報推導，並在論點裡講明。

    刻意不手寫論點。手寫會讓它讀起來跟已實查的一樣有把握，
    而它們的證據強度差一個量級。規則寫在這裡，讀者看得到推導過程。

    用橫斷面百分位而非原始值 —— 這是第一版踩到的坑：
    原本用 eps_yoy × 0.02 再截斷在 ±0.03，而 2026 年多數個股正從
    2025 的低基期回升，年增動輒數百 %，所有人都頂到截斷值。
    結果 13/31 檔的 P漲 都黏在上限 0.60，那 13 檔之間就只剩波動度在排序。
    一個對 42% 樣本都生效的上限不是護欄，是模型本身。
    改用百分位後，調整量自然分散在橫斷面上，不會集體飽和。

    金控沒有毛利率，第一版因此把它們整組排除，少掉 0050 約 7% 的權重。
    改為依可得欄位動態組合：有毛利率就用，沒有就只用 EPS 與估值。
    """
    out = []
    d = f.copy()

    def pct(col, invert=False):
        """轉成 [-0.5, +0.5] 的橫斷面百分位；invert=True 表示越小越好。"""
        v = d[col]
        if v.notna().sum() < 5:
            return pd.Series(0.0, index=d.index)
        r = v.rank(pct=True) - 0.5
        return (-r if invert else r).fillna(0.0)

    # 四個訊號，權重依「有多難造假、多慢反轉」給
    w = {"gm_chg_4q": 0.06,    # 毛利率趨勢：最慢反轉、最難造假
         "roe_ttm":   0.04,    # 資本效率：一年尺度上比單季獲利更能分辨體質
         "eps_yoy":   0.03,    # 已實現的獲利動能
         "PER":       0.04,    # 高估值＝期望已被計入，一年尺度是風險
         "rev_cagr_3y": 0.03}  # 三年複合成長，濾掉單季雜訊
    adj = (pct("gm_chg_4q") * w["gm_chg_4q"]
           + pct("roe_ttm") * w["roe_ttm"]
           + pct("eps_yoy") * w["eps_yoy"]
           + pct("PER", invert=True) * w["PER"]
           + pct("rev_cagr_3y") * w["rev_cagr_3y"])

    for i, r in d.iterrows():
        if pd.isna(r.gm_chg_4q) and pd.isna(r.eps_yoy):
            continue                                   # 財報不足，不做判斷
        p = float(np.clip(ANCHOR + adj.loc[i], 0.44, 0.64))
        why = []
        for c, lbl, fmt in (("gm_chg_4q", "毛利率較四季前", "pp"),
                            ("roe_ttm", "ROE", "%"),
                            ("eps_yoy", "EPS TTM 年增", "%"),
                            ("rev_cagr_3y", "三年營收 CAGR", "%")):
            v = r[c]
            if not pd.isna(v):
                q = int((d[c].rank(pct=True).loc[i]) * 100)
                unit = f"{v*100:+.1f}pp" if fmt == "pp" else f"{v*100:+.0f}%"
                why.append(f"{lbl}{unit}（同業第 {q} 百分位）")
        if not pd.isna(r.PER) and r.PER > 0:
            q = int((1 - d["PER"].rank(pct=True).loc[i]) * 100)
            why.append(f"PER {r.PER:.1f}（便宜度第 {q} 百分位）")

        gm = r.gross_margin
        cps = []
        if not pd.isna(gm):
            cps.append(CP(f"毛利率 ≥{(gm-0.03)*100:.1f}%（獲利結構未惡化）",
                          "gross_margin", ">=", round(float(gm - 0.03), 4)))
        if not pd.isna(r.eps_yoy):
            cps.append(CP("EPS TTM 年增維持正值", "eps_yoy", ">=", 0.0))
        cps.append(CP("月營收 TTM 年增維持正值", "rev_yoy_ttm", ">=", 0.0))
        out.append((r.code, round(p, 3), round((p - ANCHOR) * 1.2, 3), "low",
                    "RULE:未實查公司前瞻計畫，本判斷由財報規則推導（橫斷面百分位）："
                    + "、".join(why) + "。證據強度低於已實查標的，信心一律標低。",
                    cps))
    return out


def build() -> dict:
    from features import fundamentals as F
    p = pd.read_parquet(Path(__file__).resolve().parent.parent
                        / "data/features/panel.parquet")
    p = p[p.market == "TW"].sort_values(["code", "date"])
    p["r"] = p.groupby("code")["close"].pct_change()
    vol60 = p.groupby("code")["r"].apply(lambda s: s.tail(60).std())

    b = pd.read_parquet(Path(__file__).resolve().parent.parent / "data/briefing.parquet")
    b = b[b.market == "TW"][["code", "名稱", "PER"]]
    f = F.as_of("2026-09-17").merge(b, on="code", how="inner")

    # 金控明確排除，不讓它們靜默消失。
    # 理由一：損益表結構不同 —— 沒有毛利率這個概念（用 NetInterestIncome／
    #        NetNonInterestIncome），而毛利率是本規則權重最高的訊號。
    #        拿毛利率規則去判斷銀行是錯的，不是保守。
    # 理由二：FinMind 的金控 EPS 只到 2025Q4，非金融已到 2026Q2 ——
    #        用落後兩季的獲利去做一年期判斷，本身就是失真。
    # 正解是另建銀行專用特徵（NIM／ROE／淨值成長／信用成本），值得單獨一輪。
    fin_codes = sorted(c for c in f.code if c.startswith("28"))
    f = f[~f.code.isin(fin_codes)]

    # 臨床階段生技同樣排除，理由與金控類似但不同：它們的財報是真的，
    # 只是毫無判斷價值 —— 6446（藥華藥）營收 0.24 億、研發費用 3.55 億，
    # 營益率 −1488%；6919（康霈）2026Q2 營益率 −7378%。
    # 這類公司一年後的股價取決於臨床試驗讀出與法規審查，不是毛利率趨勢。
    # 拿財報規則去排序它們，產生的數字看起來有根據，實際上毫無意義。
    # 判準：營益率長期低於 −300%（正常經營的公司不會出現這種數字）。
    clinical = sorted(f.loc[f.op_margin < -3.0, "code"].unique())
    f = f[~f.code.isin(clinical)]

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
        # 分位數走對數常態。常態的左尾會延伸到 −∞，而報酬不可能低於 −100%；
        # 一年期又高波動時會算出負的股價（實測南電 −316.5 元）。見 models/quantiles.py
        sig = float(v) * math.sqrt(HORIZON)
        q10, q90 = quantiles(ev, sig)
        researched = not why.startswith("RULE:")
        why_clean = why.removeprefix("RULE:")
        th, facts = THESIS.get(code, (None, None))
        if th is None:
            th = "未實查前瞻計畫；僅依財報橫斷面位置推導"
            facts = [why_clean]
        # 否證條件就是檢查點 —— 它們已經是機器可讀的形式，
        # 不另外寫一段人話版本，避免兩邊漂移。
        falsifier = "；".join(c["claim"] for c in cps) + " —— 任一前提被推翻即論點動搖"
        out.append({
            "code": code, "prob_up": round(p_up, 4),
            "stance": "bullish" if p_up * (base * (1 + skew)) + (1 - p_up) * (-base * (1 - skew)) >= 0 else "bearish",
            "thesis": th, "facts": facts, "inference": why_clean,
            "falsifier": falsifier, "researched": researched,
            "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
            "exp_ret": round(ev, 6),
            "ret_q10": round(q10, 6), "ret_q90": round(q90, 6),
            "conviction": conf, "rationale": why_clean, "checkpoints": cps,
        })
    out.sort(key=lambda x: -x["exp_ret"])
    return {
        "as_of": AS_OF, "horizon": HORIZON, "market": "TW",
        "version": "1y-1.0.0",
        "market_context": (
            "一年期判斷的重心與 5／20 日完全不同。台積電一檔佔 0050 權重 51.5%、"
            "前 16 檔佔 80.5% —— 一年期只要台積電錯了，整個指數判斷就錯了。"
            "已實查前瞻計畫的標的合計約佔權重 70%（2330／2454／2308 及 AI 伺服器"
            "供應鏈的產業層級），其餘僅由財報規則推導、信心一律標低。"
            "本批最重要的共同風險：AI 伺服器鏈（2317／2382／3231／3037／3189）"
            "全繫於北美雲端服務商的資本支出，那是單一共同因子，"
            "不是五個獨立賭注 —— 排序上看似分散，實際不是。"
            "每一檔都附檢查點，前提每月／每季自動對帳，論點壞掉當下就知道，不必等到期。"
            "\n\n已知缺口：金控股（約佔 0050 權重 8%）不在本批判斷內。它們的損益表沒有"
            "毛利率這個概念，而毛利率是本規則權重最高的訊號；且 FinMind 的金控 EPS "
            "只到 2025Q4、比非金融落後兩季。拿毛利率規則去判斷銀行是錯的，"
            "要另建 NIM／ROE／淨值成長那套特徵才算數。"
            "臨床階段生技（如 6446、6919）也排除：它們的營益率是 −1488%、−7378%，"
            "數字是真的但沒有判斷價值 —— 一年後的股價取決於臨床讀出與法規審查，"
            "不是毛利率趨勢。"),
        "judgments": out,
    }


if __name__ == "__main__":
    d = build()
    out = Path(__file__).resolve().parent / f"judgment_1y_{AS_OF}.json"
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"一年期判斷：{len(d['judgments'])} 檔 → {out.name}")
    n_cp = sum(len(j["checkpoints"]) for j in d["judgments"])
    n_res = sum(j["researched"] for j in d["judgments"])
    print(f"  檢查點 {n_cp} 條；已實查前瞻計畫 {n_res} 檔、規則推導 {len(d['judgments'])-n_res} 檔")
    print(f"\n  {'代號':<7}{'P漲':>6}{'漲幅':>9}{'跌幅':>9}{'期望值':>9}  信心")
    for j in d["judgments"][:8]:
        print(f"  {j['code']:<7}{j['prob_up']*100:>5.0f}%{j['up_magnitude']*100:>+8.1f}%"
              f"{j['dn_magnitude']*100:>+8.1f}%{j['exp_ret']*100:>+8.2f}%  {j['conviction']}")
