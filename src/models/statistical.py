"""統計／機器學習模型，嚴格 walk-forward。

最容易犯的錯（也是讓回測績效虛假亮眼的主因）：
  用 as_of 當天的資料訓練。但標籤需要 horizon 天後才實現，
  所以訓練資料的截止日必須是 as_of - horizon 個交易日，不是 as_of。
  下面的 train_cutoff 就是在守這條線。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features.build import FEATURE_COLS, build_all, labels_all
from config import MIN_TRAIN_ROWS

VERSION = "1.0.0"

def _training_set(panel: pd.DataFrame, as_of, horizon: int,
                  lookback_days: int = 400, stride: int = 2):
    """組訓練集：只取標籤已在 as_of 之前實現的樣本。

    cutoff 是這整個模組最關鍵的一行。標籤要 horizon 個交易日後才知道，
    所以訓練樣本的 as_of 最晚只能到 dates[-(horizon+1)]。
    若誤用 as_of 當天，模型等於偷看了答案，回測會漂亮得不真實。
    """
    as_of = pd.Timestamp(as_of)
    hist = panel[panel["date"] <= as_of]
    dates = sorted(hist["date"].unique())
    if len(dates) < horizon + 80:
        return pd.DataFrame(), pd.Series(dtype=int)
    cutoff = dates[-(horizon + 1)]

    feats = build_all(hist)
    labs = labels_all(hist, horizon)
    m = feats.merge(labs[["as_of", "code", "y"]], on=["as_of", "code"], how="inner")
    m = m[m["as_of"] <= cutoff]
    keep = sorted(m["as_of"].unique())[-lookback_days:][::stride]
    m = m[m["as_of"].isin(keep)].dropna(subset=["y"])
    if m.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    return m[FEATURE_COLS].reset_index(drop=True), m["y"].reset_index(drop=True)


def _fit(kind: str, X, y):
    if kind == "logit":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=0.3)).fit(X, y)
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingClassifier(
                             max_depth=3, max_iter=180, learning_rate=0.05,
                             l2_regularization=1.0, random_state=42)).fit(X, y)

def predict(feats: pd.DataFrame, horizon: int, as_of, panel: pd.DataFrame,
            kind: str = "logit") -> pd.DataFrame:
    X, y = _training_set(panel, as_of, horizon)
    if len(X) < MIN_TRAIN_ROWS or y.nunique() < 2:
        return pd.DataFrame()   # 資料不足時直接不出手，勝過硬猜
    model = _fit(kind, X, y)
    p = model.predict_proba(feats[FEATURE_COLS])[:, 1]
    base = float(y.mean())
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": p,
        "direction": np.where(p >= 0.5, 1, -1),
        "rationale": [f"{kind}: 訓練 {len(X):,} 筆（基本率 {base:.1%}）" for _ in p],
    })

def predict_logit(feats, horizon, as_of, panel):
    return predict(feats, horizon, as_of, panel, "logit")

def predict_gbdt(feats, horizon, as_of, panel):
    return predict(feats, horizon, as_of, panel, "gbdt")

ALL = {"stat_logit": predict_logit, "stat_gbdt": predict_gbdt}
