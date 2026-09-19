"""一年期規則推導的 P漲。建置與每日滾動共用同一份，避免兩邊漂移。

為什麼用橫斷面百分位而非原始值：
  第一版用 eps_yoy × 0.02 再截斷在 ±0.03。2026 年多數個股正從 2025 的
  低基期回升，年增動輒數百 %，全部頂到截斷值 —— 13/31 檔的 P漲 黏在
  上限 0.60，那 13 檔之間就只剩波動度在排序。
  一個對 42% 樣本都生效的上限不是護欄，是模型本身。
  百分位讓調整量自然分散在橫斷面上，不會集體飽和。

權重依「多難造假、多慢反轉」給，不是憑感覺排：
  毛利率趨勢最高 —— 它反映產品組合與定價權，一季一季慢慢動，難以粉飾。
  ROE 次之 —— 一年尺度上比單季獲利更能分辨體質。
  EPS 年增再次 —— 已實現但容易受基期與一次性項目干擾。
  估值扣分 —— 高本益比代表期望已被計入，在一年尺度上是風險不是優勢。

週期位置修正（2026-09-19 新增，VERSION 1.1.0）：
  上一版全部是**橫斷面**百分位，沒有任何**時間序列**參照，因此對
  「這是景氣循環的哪個位置」沒有辨識力。實例：華邦電毛利率 66.2%
  （自身歷史長期 20–35%）、EPS 年增 +1232%，拿到橫斷面第 98／第 100 百分位
  而被排到台股一年期最前段（P漲 0.621）；同一天手寫的美光論點講的正是
  這組數字的反面（「本益比最低的時候最貴，因為分母是週期高點的獲利」，
  P漲 0.48）。兩檔同屬一個記憶體循環卻被判到兩端，**差別只在覆蓋率不在判斷**。

  修法：用 `gm_self_pct`（當期毛利率相對該公司自身歷史分布的百分位）
  去調整三個「在週期高點會機械性衝頂」的訊號 —— 毛利率變化、EPS 年增、
  以及本益比那一項（峰值獲利會讓 PER 看起來便宜，那正是陷阱本身）。
      cyc = clip((gm_self_pct − 0.75) / 0.25, 0, 1)   # ≤75 百分位為 0，100 百分位為 1
      這三個訊號的貢獻 × (1 − cyc)                      # 週期高點時衰減到 0
      再整體扣掉 CYC_PENALTY × cyc                      # 均值回歸本身的代價
  `roe_ttm` 與 `rev_cagr_3y` 不調整：前者是水準、後者是三年趨勢，
  對單季的循環位置不敏感。

  **為什麼是「衰減到 0 再扣分」而不是「把權重乘上負數」**：
  第一版寫成 mult = 1 − 1.5×cyc（極端時 −0.5），結果讓
  「位於自身毛利率高點、但橫斷面成長排名偏低」的公司拿到**加分**
  （實測川湖 +0.020、致茂 +0.022）—— 因為負的貢獻被乘上負數就變成正的。
  那是反的：位於自身高點不該是任何形式的優勢。
  衰減到 0 表達的是「這個讀數在週期高點沒有資訊」，
  扣分表達的是「均值回歸本身是個負面因子」，兩件事分開講才不會互相污染。

  ⚠️ **證據狀態要說清楚，這一項的兩半強弱不同**（2026-09-19 實測）：
    機制成立：Spearman(gm_self_pct, 後四季毛利率變化) = **−0.143**
              （台股 52 檔、1,427 個季度觀測；≥0.95 者後四季 +0.32pp、
                <0.95 者 +0.45pp）—— 毛利率相對自身歷史確實均值回歸。
    報酬無證據：自身百分位 ≥0.95 的後 250 日報酬中位 +11.5%，
              <0.95 者 +13.8%，Spearman 僅 +0.041（448 個重疊觀測、
              獨立樣本遠少於此）。**這個修正沒有被證明能提高報酬預測力。**
  之所以仍然採用：它修的是一個**內部矛盾**（同一組事實，有人讀過就看空、
  沒人讀過就看多），而不是在追求一個回測分數。誠實的說法是
  「讓規則與已實查論點對同一組事實給出一致的方向」，不是「這樣比較準」。

  美股 `gm_self_pct` 目前全是 NaN（yfinance 一次只給 5–7 季，不足 8 季），
  修正自動不生效 —— 用 5 季的歷史講「相對自身歷史的極值」沒有意義。
  等 `data/raw/us_fund/` 累積夠了會自動開始作用。
"""
from __future__ import annotations
import numpy as np, pandas as pd

VERSION = "1.1.0"    # 改這個字串會讓 roll_1y 的基礎雜湊失配，因而合法地觸發
                     # 一次「規則改版」重算 —— 且帳本會記成規則改版，
                     # 不會偽裝成「財報已更新」（那會是一句假話）。

ANCHOR = 0.55        # 中性錨點。見 build_1y 的說明：樣本內 79% 為正是
                     # 兩年單一多頭，不可用；改用大盤 65–70% 再往下調整。
WEIGHTS = {"gm_chg_4q": 0.06, "roe_ttm": 0.04, "eps_yoy": 0.03,
           "PER": 0.04, "rev_cagr_3y": 0.03}
FLOOR, CAP = 0.44, 0.64

# 在週期高點會機械性衝頂的訊號 —— 只有這三個受週期位置修正影響。
CYCLICAL = ("gm_chg_4q", "eps_yoy", "PER")
CYC_LO, CYC_HI = 0.75, 1.0
CYC_PENALTY = 0.05   # 位於自身毛利率分布頂端時，額外扣掉的 P漲


def prob_up(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """回傳 (P漲, 說明文字)，索引與 d 相同。d 需含 WEIGHTS 的各欄。"""
    def pct(col, invert=False):
        if col not in d.columns or d[col].notna().sum() < 5:
            return pd.Series(0.0, index=d.index)
        r = d[col].rank(pct=True) - 0.5
        return (-r if invert else r).fillna(0.0)

    # 週期位置：當期毛利率落在自身歷史第幾百分位。缺值（季數不足）→ 不修正。
    if "gm_self_pct" in d.columns:
        cyc = ((d["gm_self_pct"] - CYC_LO) / (CYC_HI - CYC_LO)).clip(0, 1).fillna(0.0)
    else:
        cyc = pd.Series(0.0, index=d.index)
    keep = 1.0 - cyc                     # 週期高點時，成長與估值類訊號衰減到 0

    adj = sum(pct(c, invert=(c == "PER")) * w * (keep if c in CYCLICAL else 1.0)
              for c, w in WEIGHTS.items())
    p = (ANCHOR + adj - CYC_PENALTY * cyc).clip(FLOOR, CAP)

    why = []
    for i in d.index:
        parts = []
        for c, lbl, unit in (("gm_chg_4q", "毛利率較四季前", "pp"),
                             ("roe_ttm", "ROE", "%"),
                             ("eps_yoy", "EPS TTM 年增", "%"),
                             ("rev_cagr_3y", "三年營收 CAGR", "%")):
            if c not in d.columns or pd.isna(d.loc[i, c]):
                continue
            v = d.loc[i, c]
            q = int(d[c].rank(pct=True).loc[i] * 100)
            txt = f"{v*100:+.1f}pp" if unit == "pp" else f"{v*100:+.0f}%"
            parts.append(f"{lbl}{txt}（同業第 {q} 百分位）")
        if "PER" in d.columns and pd.notna(d.loc[i, "PER"]) and d.loc[i, "PER"] > 0:
            q = int((1 - d["PER"].rank(pct=True).loc[i]) * 100)
            parts.append(f"PER {d.loc[i,'PER']:.1f}（便宜度第 {q} 百分位）")
        tail = ""
        if "gm_self_pct" in d.columns and pd.notna(d.loc[i, "gm_self_pct"]):
            sp = float(d.loc[i, "gm_self_pct"])
            if cyc.loc[i] > 0:
                tail = (f"　⚠ 當期毛利率位於**自身歷史第 {sp*100:.0f} 百分位**："
                        f"成長與估值類訊號已衰減至原權重的 {keep.loc[i]*100:.0f}%，"
                        f"並扣除 {CYC_PENALTY * cyc.loc[i]:.3f} 的週期位置代價 —— "
                        f"週期高點的成長率與低本益比都是分母造成的，不是體質。")
        why.append("未實查公司前瞻計畫，本判斷由財報規則推導（橫斷面百分位）："
                   + "、".join(parts) + "。證據強度低於已實查標的，信心一律標低。"
                   + tail)
    return p.round(3), pd.Series(why, index=d.index)
