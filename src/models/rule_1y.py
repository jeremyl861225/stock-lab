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
"""
from __future__ import annotations
import numpy as np, pandas as pd

ANCHOR = 0.55        # 中性錨點。見 build_1y 的說明：樣本內 79% 為正是
                     # 兩年單一多頭，不可用；改用大盤 65–70% 再往下調整。
WEIGHTS = {"gm_chg_4q": 0.06, "roe_ttm": 0.04, "eps_yoy": 0.03,
           "PER": 0.04, "rev_cagr_3y": 0.03}
FLOOR, CAP = 0.44, 0.64


def prob_up(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """回傳 (P漲, 說明文字)，索引與 d 相同。d 需含 WEIGHTS 的各欄。"""
    def pct(col, invert=False):
        if col not in d.columns or d[col].notna().sum() < 5:
            return pd.Series(0.0, index=d.index)
        r = d[col].rank(pct=True) - 0.5
        return (-r if invert else r).fillna(0.0)

    adj = sum(pct(c, invert=(c == "PER")) * w for c, w in WEIGHTS.items())
    p = (ANCHOR + adj).clip(FLOOR, CAP)

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
        why.append("未實查公司前瞻計畫，本判斷由財報規則推導（橫斷面百分位）："
                   + "、".join(parts) + "。證據強度低於已實查標的，信心一律標低。")
    return p.round(3), pd.Series(why, index=d.index)
