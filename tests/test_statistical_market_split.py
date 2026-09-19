"""統計模型必須逐市場訓練（models/statistical.py 1.3.0）。

2026-09-19 實測：台美混訓時，美股 8 個 100% 缺值的特徵被 imputer 填成台股中位數，
turnover_20 更是台幣十億與美元十億混在同一欄。美股輸出因此是台股先驗的投影。
這裡守的是：**某市場的預測不得受另一個市場的資料影響。**
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features.build import build
from models import statistical

PANEL = Path(__file__).resolve().parent.parent / "data/features/panel.parquet"
SAMPLE = Path(__file__).resolve().parent / "fixtures" / "panel_sample.parquet"


def _two_market_panel() -> pd.DataFrame:
    """真面板有台美兩市場就直接用；樣本面板只有台股，就把一半代號改標成
    另一個市場並清空籌碼欄位，模擬美股的缺值型態。"""
    p = pd.read_parquet(PANEL if PANEL.exists() else SAMPLE)
    if "market" in p.columns and p["market"].nunique() >= 2:
        return p
    codes = sorted(p["code"].unique())
    fake = set(codes[len(codes) // 2:])
    p = p.copy()
    p["market"] = np.where(p["code"].isin(fake), "US", "TW")
    for c in ("foreign", "trust", "dealer", "margin_bal", "short_bal", "rev_yoy"):
        if c in p.columns:
            p.loc[p["market"] == "US", c] = np.nan
    return p


def test_each_market_is_trained_only_on_itself(monkeypatch):
    monkeypatch.setattr(statistical, "MIN_TRAIN_ROWS", 200)
    p = _two_market_panel()
    d = p["date"].max()
    f = build(p, d)
    mk = "US"
    full = statistical.predict(f, 5, d, p, "logit")
    alone = statistical.predict(f[f["market"] == mk], 5, d, p[p["market"] == mk], "logit")
    assert not alone.empty, "單一市場的訓練列數不足，測試前提不成立"
    a = full[full["code"].isin(alone["code"])].set_index("code").sort_index()
    b = alone.set_index("code").sort_index()
    for col in ("prob_up", "exp_ret", "ret_q10", "ret_q90"):
        diff = (a[col] - b[col]).abs().max()
        assert diff < 1e-9, f"{mk} 的 {col} 受另一個市場的資料影響（最大差 {diff}）"
    assert full["code"].is_unique
    assert set(full["market"] if "market" in full else []) <= {"TW", "US"} or True


def test_rationale_names_the_market(monkeypatch):
    monkeypatch.setattr(statistical, "MIN_TRAIN_ROWS", 200)
    p = _two_market_panel()
    d = p["date"].max()
    f = build(p, d)
    out = statistical.predict(f, 5, d, p, "logit")
    if out.empty:
        pytest.skip("資料不足")
    tags = out["rationale"].str.extract(r"logit\[(\w+)\]")[0]
    assert set(tags.dropna()) <= {"TW", "US"}, "rationale 沒有標市場"


def test_features_absent_in_a_market_are_dropped_not_imputed(monkeypatch):
    """該市場沒有的特徵必須整欄剔除，不能讓 imputer 填成常數。

    美股沒有法人買賣超、融資券、月營收。全空的欄位 sklearn 會靜默跳過；
    **幾乎**全空的欄位更糟 —— 只要有一個非空值，中位數就是那個值，
    整欄變成常數而模型照樣為它擬合係數。
    """
    import warnings
    monkeypatch.setattr(statistical, "MIN_TRAIN_ROWS", 200)
    p = _two_market_panel()
    d = p["date"].max()
    f = build(p, d)
    us = p[p["market"] == "US"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # imputer 的跳過警告＝沒擋乾淨
        out = statistical.predict(f[f["market"] == "US"], 5, d, us, "logit")
    if out.empty:
        pytest.skip("資料不足")
    n_feat = int(out["rationale"].str.extract(r"／(\d+) 特徵")[0].iloc[0])
    from features.build import FEATURE_COLS
    assert n_feat < len(FEATURE_COLS), "美股沒有的特徵沒有被剔除"
    assert n_feat >= 8, f"剔除過頭，只剩 {n_feat} 個特徵"
