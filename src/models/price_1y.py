# -*- coding: utf-8 -*-
"""一年期（h=250）定價：由 P漲 與 σ 導出**同一個**對數常態分布的全部數字。

為什麼一年期不能沿用 5／20 日那套「兩點模型＋對數常態區間」：
  5／20 日的期望值走 base×(2p−1+skew)，區間走 σ=vol×√h 的對數常態，
  兩者對 P漲 的隱含值差 0.01–0.03，可以忍。
  但 h=250、σ 到 0.6–1.0 時，對數常態的 σ²/2 拖累達 0.2–0.5，中位數轉負：
  2026-09-19 實測台股 38 檔一年期，手寫 P漲 均值 0.552，
  而它們自己的 q10/q90 所隱含的 P漲 均值只有 0.406（最大差 0.206）。
  南電 8046：P漲 0.53、區間 [−83%, +140%]、隱含 P漲 0.33，
  面板目標價 +85%、保守價 −83% —— 那兩個數字是波動度的讀數，不是判斷。
  Brier 用手寫的 P漲 打分、區間覆蓋率用 q10/q90 打分，兩者根本不是同一個預測。

所以一年期只有一個原始量：**P漲**（判斷給的）。其餘全部由它與 σ 導出：
  對數報酬 Y ~ N(m, σ²)，令 P(Y>0)=p  ⇒  m = σ·Φ⁻¹(p)
  中位數（面板的 exp_ret）        exp(m) − 1                     ＝第 50 百分位
  上漲情境的中位幅度（目標價）    exp(m + σ·Φ⁻¹(1 − p/2)) − 1    ＝第 (1−p/2) 百分位
  下跌情境的中位幅度              exp(m + σ·Φ⁻¹((1−p)/2)) − 1    ＝第 (1−p)/2 百分位
  分位數（保守價用 q10）          exp(m ∓ z·σ) − 1（z 同 quantiles.Z10）
  平均數（僅記錄，不用來排序）    exp(m + σ²/2) − 1

為什麼 exp_ret 在一年期是中位數而不是平均數：
  對數常態的平均數含 exp(σ²/2)，σ=0.8 時是中位數的 1.38 倍。用平均數排序，
  2026-09-18 的台股一年期會變成純粹的波動度排序（南亞科 +89%、南電 +55%、
  台積電 +15%、中華電 +1%），判斷本身被 σ 淹掉；而實證上高波動個股的
  一年平均報酬並不高（低波動異象）。中位數對 σ 只有線性相依，且
  中位數 > 0 ⟺ P漲 > 0.5，看多／看空的標籤與機率永遠一致。
  平均數仍寫進帳本（mean_ret），因為它是這個分布真正的期望值，
  結算後可以拿來檢驗對數常態在一年尺度上是否高估了右尾。

一年期**沒有獨立的 skew**：對數常態的不對稱完全由 σ 決定。原本的 skew 對規則
推導標的是 (P−0.55)×1.2，本來就是 P漲 的函數；手寫的也幾乎與 P漲 同向，
拿掉不損失資訊，卻消掉「skew 以 0.55 為中心、期望值以 0.5 為中心」造成的
「P漲 50% 卻標看空」（2026-09-19 台股每天約 4 檔）。

σ 的來源（W_SHORT）：
  vol_60 是 regime 讀數。2026-09-19 台股 vol_60 中位是自身長期波動的 1.22 倍、
  34% 的檔位超過 1.3 倍。用 16 檔五年以上的台股（1,424 組、步長 10 日）
  擬合「後 250 日實現變異 = a·vol_60² + b·vol_LR²」（NNLS）：
    a=0.474、b=0.319 → 正規化後 vol_60 權重 0.60（逐檔留一法 0.51–0.64）
    log 誤差 RMSE：只用 vol_60 0.321、只用長期 0.279、混合 0.254
    vol_60 在最高四分位時，實現／預測中位：只用 vol_60 0.91、混合 0.90
  混合把大部分的均值誤差修掉，regime 尾端約一成的高估仍在，這是已知殘餘。
  vol_LR 取最近 1,000 日；不足 250 日則不混合，退回 vol_60。
  美股只有兩年資料，無法獨立擬合，沿用台股權重。

METHOD.md §5.6 引用這裡的常數，tests/test_method_doc.py 釘住它們；
tests/test_price_1y.py 守恆等式與單調性。
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from models.quantiles import Z10

W_SHORT = 0.6              # vol_60 在變異數混合裡的權重（見上方擬合）
LR_MIN, LR_MAX = 250, 1000  # 長期波動的最少／最多觀測日數


def sigma_daily(r) -> float | None:
    """由日對數報酬序列（已按日期排序，可含 NaN）算混合後的日 σ。

    資料不足 60 日回 None；不足 LR_MIN 日則不混合，直接用 vol_60。
    """
    x = pd.Series(r, dtype="float64").dropna().to_numpy()
    if len(x) < 60:
        return None
    v60 = float(np.std(x[-60:], ddof=1))
    if not np.isfinite(v60) or v60 <= 0:
        return None
    if len(x) < LR_MIN:
        return v60
    vlr = float(np.std(x[-LR_MAX:], ddof=1))
    if not np.isfinite(vlr) or vlr <= 0:
        return v60
    return math.sqrt(W_SHORT * v60 ** 2 + (1.0 - W_SHORT) * vlr ** 2)


def price(p: float, sigma_h: float, z: float = Z10) -> dict:
    """p：P(h 日後收盤高於今日)；sigma_h：該期間對數報酬標準差（日 σ × √h）。

    回傳的每一個數字都是同一個 N(m, σ²) 的分位數或矩，所以恆有
      Φ(m/σ) == p、dn < median < up、median > 0 ⟺ p > 0.5、
      p·up_mean + (1−p)·dn_mean == mean。

    ⚠️ `q10 < dn` **只在 p < 0.826 時成立**，這不是 bug：
      q10 用 Z10 = 1.36（比常態的 10% 分位 1.2816 寬，見 quantiles.py），
      落在約第 8.7 百分位；而 dn 落在第 (1−p)/2 百分位。
      p > 1 − 2×0.087 ≈ 0.826 時 (1−p)/2 < 0.087，兩者交叉，
      「保守價」會高於「下跌情境的中位數」。
      實務上不會遇到：規則推導的 P漲 上限是 0.64（rule_1y.CAP），
      手寫的一年期 P漲 目前分布在 0.469–0.631。
      但若哪天有人寫下 p ≥ 0.83，面板上那兩個價位的大小關係會反過來 ——
      到時候要改的是顯示，不是這裡的數學。
    """
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    s = float(sigma_h)
    if not (np.isfinite(s) and s > 0):
        raise ValueError(f"sigma_h 必須為正：{sigma_h}")
    m = s * float(ndtri(p))
    mean = math.exp(m + s * s / 2.0)
    a = float(ndtr(m / s + s))                 # Φ(m/σ + σ)
    return {
        "median": math.exp(m) - 1.0,
        "mean": mean - 1.0,
        "up": math.exp(m + s * float(ndtri(1.0 - p / 2.0))) - 1.0,
        "dn": math.exp(m + s * float(ndtri((1.0 - p) / 2.0))) - 1.0,
        "up_mean": mean * a / p - 1.0,
        "dn_mean": mean * (1.0 - a) / (1.0 - p) - 1.0,
        "q10": math.exp(m - z * s) - 1.0,
        "q90": math.exp(m + z * s) - 1.0,
        # 四分位：面板一年期顯示「五成區間」用的。
        # 不是為了讓區間看起來窄一點 —— q25–q75 的寬度只比 q10–q90 少一成半
        # （2026-09-19 實測：中位 1.02 倍收盤 vs 1.18 倍）。
        # 換成它是因為「一年後有一半機率落在這裡」是一句說得清楚的話，
        # 而「目標價／保守價」會被讀成可執行的價位，那是誤導。
        "q25": math.exp(m - 0.6744897501960817 * s) - 1.0,
        "q75": math.exp(m + 0.6744897501960817 * s) - 1.0,
        "m": m, "sigma": s,
    }


def implied_prob_up(q10: float, q90: float, z: float = Z10) -> float:
    """由 (q10, q90) 反推該對數常態分布隱含的 P漲：m 是兩個分位數的中點。

    2026-09-19 抓出不一致用的是「由平均數反推 m」的版本；改成由分位數中點
    反推之後，對舊帳本（exp_ret 為兩點模型）與新帳本（exp_ret 為中位數）都適用。
    """
    s = (math.log1p(q90) - math.log1p(q10)) / (2.0 * z)
    m = (math.log1p(q90) + math.log1p(q10)) / 2.0
    return float(ndtr(m / s))
