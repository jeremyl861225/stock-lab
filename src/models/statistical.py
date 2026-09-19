"""統計／機器學習模型，嚴格 walk-forward，輸出報酬分布而非只有方向。

為什麼不能只預測方向：
  P(up)=0.60 的標的，若上漲時平均 +1%、下跌時平均 -3%，
  期望報酬是 0.6×1% − 0.4×3% = −0.6% —— 方向準確率漂亮，實際賠錢。
  高波動股票特別容易出現這種型態，只看方向的計分完全抓不到。
所以每個模型都必須同時給出：
  prob_up  方向機率
  exp_ret  期望報酬（機率與幅度加權後的結果，真正該用來排序的數字）
  q10/q90  報酬的下檔與上檔（不確定性有多大、風險是否對稱）

最關鍵的一行仍是 cutoff：標籤要 horizon 個交易日後才實現，
訓練樣本的 as_of 最晚只能到 dates[-(horizon+1)]，否則就是偷看答案。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.quantiles import quantiles
from features.build import FEATURE_COLS, build_all, labels_all
from config import MIN_TRAIN_ROWS

VERSION = "1.3.0"     # 1.3.0：分市場訓練（見 predict 的說明）

# 某個特徵在該市場訓練集裡的最低覆蓋率。低於此值就整欄剔除，不進模型。
# 為什麼要明確剔除，而不是交給 imputer：
#   全空的欄位 sklearn 的 SimpleImputer 會「跳過」（連同警告），行為正確但是靜默的；
#   更危險的是**幾乎**全空的欄位 —— 只要有一個非空值，中位數就等於那個值，
#   於是整欄變成一個常數，模型會把它當成真實特徵去擬合係數。
#   美股的 foreign_5／trust_5／rev_yoy／margin_chg_*／short_ratio 全部 100% 缺值，
#   剔除後美股由 19 個特徵降為 12 個 —— 那 12 個才是美股真的有的資訊。
MIN_FEATURE_COVERAGE = 0.2


def _training_set(panel: pd.DataFrame, as_of, horizon: int,
                  lookback_days: int = 400, stride: int = 2):
    """回傳 (X, y_dir, y_ret)。只取標籤已在 as_of 之前實現的樣本。

    panel 應已限定單一市場（見 predict）：cutoff 用的是該市場自己的交易日曆，
    台美混在一起時 dates[-(h+1)] 可能落在只有一個市場開盤的日子。
    """
    as_of = pd.Timestamp(as_of)
    hist = panel[panel["date"] <= as_of]
    dates = sorted(hist["date"].unique())
    if len(dates) < horizon + 80:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=float)
    cutoff = dates[-(horizon + 1)]          # ← 偷看未來的唯一防線

    feats = build_all(hist)
    labs = labels_all(hist, horizon)
    m = feats.merge(labs[["as_of", "code", "y", "fwd_ret"]], on=["as_of", "code"], how="inner")
    m = m[m["as_of"] <= cutoff]
    keep = sorted(m["as_of"].unique())[-lookback_days:][::stride]
    m = m[m["as_of"].isin(keep)].dropna(subset=["y", "fwd_ret"])
    if m.empty:
        return pd.DataFrame(), pd.Series(dtype=int), pd.Series(dtype=float)
    return (m[FEATURE_COLS].reset_index(drop=True),
            m["y"].reset_index(drop=True),
            m["fwd_ret"].reset_index(drop=True))


def _clf(kind: str, X, y):
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.3)).fit(X, y)
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(
                             max_depth=3, max_iter=180, learning_rate=0.05,
                             l2_regularization=1.0, random_state=42)).fit(X, y)


def _reg(kind: str, X, y, quantile: float | None = None):
    """quantile=None 時預測條件均值；否則預測該分位數。"""
    if kind == "logit":     # 線性家族用 Ridge，分位數以常態假設近似
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             Ridge(alpha=3.0)).fit(X, y)
    if quantile is None:
        return make_pipeline(SimpleImputer(strategy="median"),
                             HistGradientBoostingRegressor(
                                 max_depth=3, max_iter=150, learning_rate=0.05,
                                 l2_regularization=1.0, random_state=42)).fit(X, y)
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(
                             loss="quantile", quantile=quantile, max_depth=3,
                             max_iter=150, learning_rate=0.05,
                             l2_regularization=1.0, random_state=42)).fit(X, y)


def predict(feats: pd.DataFrame, horizon: int, as_of, panel: pd.DataFrame,
            kind: str = "logit") -> pd.DataFrame:
    """逐市場各訓練一個模型，再把結果併回來。

    為什麼不能台美混訓（2026-09-19 實測）：
      美股沒有法人買賣超、融資券、月營收，8 個特徵 100% 缺值；混訓時 imputer
      的中位數來自台股列（台股列數是美股的 1.7 倍），於是每一檔美股都被餵進
      「台股中位數的月營收年增、外資買超、融資變化」—— 而 rev_yoy 是 Fama-MacBeth
      t=5.26 的強特徵。turnover_20 更是台幣十億與美元十億混在同一欄（差 30 倍幣值）。
      結果 stat 模型的美股輸出大半是台股先驗的投影。
    分開訓練之後，缺值的填補、幣別、交易日曆、基本率全部各自歸各自的市場。
    """
    if "market" in feats.columns and "market" in panel.columns:
        parts = []
        for mk, fm in feats.groupby("market", sort=False):
            out = _predict_one(fm, horizon, as_of, panel[panel["market"] == mk], kind, mk)
            if not out.empty:
                parts.append(out)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return _predict_one(feats, horizon, as_of, panel, kind, None)


def _predict_one(feats: pd.DataFrame, horizon: int, as_of, panel: pd.DataFrame,
                 kind: str, market: str | None) -> pd.DataFrame:
    X, y_dir, y_ret = _training_set(panel, as_of, horizon)
    if len(X) < MIN_TRAIN_ROWS or y_dir.nunique() < 2:
        return pd.DataFrame()          # 資料不足就不出手，勝過硬猜

    cols = [c for c in FEATURE_COLS
            if X[c].notna().mean() >= MIN_FEATURE_COVERAGE]
    if not cols:
        return pd.DataFrame()
    X = X[cols]
    Xp = feats[cols]
    prob = _clf(kind, X, y_dir).predict_proba(Xp)[:, 1]
    exp_ret = _reg(kind, X, y_ret).predict(Xp) if kind != "logit" else None

    if kind == "logit":
        # 線性家族不做分位數回歸，用殘差標準差的常態近似 ——
        # 但必須按各股波動度縮放。原本用單一純量 σ，導致中華電（日波動 0.48%）
        # 與大立光（5.91%）拿到完全相同的 35.2% 區間，
        # 條件覆蓋率從最低波動組 99% 到最高波動組 65%（名目 80%）。
        reg = _reg(kind, X, y_ret)                      # 只擬合一次，原本擬合了兩次
        exp_ret = reg.predict(Xp)
        resid = y_ret - reg.predict(X)
        # 殘差除以各自的波動度 → 得到無單位的殘差尺度，再乘回每檔自己的波動度
        train_vol = X["vol_20"].replace(0, np.nan)
        z = (resid / (train_vol * np.sqrt(horizon))).replace([np.inf, -np.inf], np.nan).dropna()
        k = float(z.std()) if len(z) > 30 and z.std() > 0 else 1.0
        sig = feats["vol_20"].fillna(feats["vol_20"].median()).to_numpy() * np.sqrt(horizon) * k
        # 對數常態，理由同 models/quantiles.py：常態左尾會越過 −100%
        q10, q90 = quantiles(exp_ret, sig, z=1.2816)
    else:
        q10 = _reg(kind, X, y_ret, 0.10).predict(Xp)
        q90 = _reg(kind, X, y_ret, 0.90).predict(Xp)
        q10, q90 = np.minimum(q10, q90), np.maximum(q10, q90)

    base = float(y_dir.mean())
    base_ret = float(y_ret.mean())
    return pd.DataFrame({
        "code": feats["code"].values,
        "prob_up": prob,
        "exp_ret": exp_ret,
        "ret_q10": q10,
        "ret_q90": q90,
        "direction": np.where(exp_ret >= 0, 1, -1),   # 方向改由期望報酬決定
        "rationale": [f"{kind}[{market or 'ALL'}]: 訓練 {len(X):,} 筆／{len(cols)} 特徵"
                      f"（基本率 {base:.1%}／平均報酬 {base_ret:+.2%}）" for _ in prob],
    })


def predict_logit(feats, horizon, as_of, panel):
    return predict(feats, horizon, as_of, panel, "logit")


def predict_gbdt(feats, horizon, as_of, panel):
    return predict(feats, horizon, as_of, panel, "gbdt")


ALL = {"stat_logit": predict_logit, "stat_gbdt": predict_gbdt}
