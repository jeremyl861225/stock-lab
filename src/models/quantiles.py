"""報酬分位數。長期間必須用對數常態，不能用常態。

為什麼：常態分佈的左尾會延伸到 −∞，而報酬率不可能低於 −100%（股價不會變負）。
期間短、波動低時這個誤差可以忽略；期間拉到一年、波動拉到 90% 就會炸開。

實測（2026-09-17 的一年期判斷，38 檔）：
  q10 = 期望值 − 1.36σ
  → 18 檔的 q10 < −90%，其中 7 檔算出負的股價
    南電 −316.5 元、創意 −914.7 元、景碩 −203.0 元
  這些數字已經上線在面板的「保守價」欄位上。

對數常態的做法：
  對數報酬 ~ N(m, σ²)，其中 σ = 日波動 × √h
  m 取成讓「簡單報酬的期望值」等於判斷給的 exp_ret：
    E[比值] = exp(m + σ²/2) = 1 + exp_ret  →  m = ln(1+exp_ret) − σ²/2
  分位數：比值 = exp(m ± z·σ)，永遠為正。

z 用 1.36 而非常態的 1.2816：
  σ 是從有限樣本估出來的，本身帶不確定性。用 t₁₉ 的 10% 分位（1.328）
  再留一點餘裕，避免區間過窄而讓「落在區間內」的比率虛高。
"""
from __future__ import annotations
import numpy as np

Z10 = 1.36          # 10%／90% 分位所用的 z（已含參數不確定性的加寬）


def quantiles(exp_ret, sigma, z: float = Z10):
    """回傳 (q10, q90)，皆為簡單報酬。sigma 是該期間的對數報酬標準差。

    exp_ret 與 sigma 可以是純量或 numpy 陣列。
    保證 q10 > −1（即價格恆正）。
    """
    ev = np.asarray(exp_ret, dtype=float)
    sd = np.asarray(sigma, dtype=float)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
    # 判斷給的 exp_ret 若 ≤ −100%，本身就已經無意義，夾到可表達的範圍
    ratio_mean = np.maximum(1.0 + ev, 1e-6)
    m = np.log(ratio_mean) - sd ** 2 / 2
    q10 = np.exp(m - z * sd) - 1.0
    q90 = np.exp(m + z * sd) - 1.0
    if np.isscalar(exp_ret) and np.isscalar(sigma):
        return float(q10), float(q90)
    return q10, q90
