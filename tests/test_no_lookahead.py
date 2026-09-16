"""系統最重要的一組測試：證明特徵沒有偷看未來。

這不是形式上的單元測試 —— 未來函數是這類系統最常見、也最致命的錯誤，
而且它的症狀是「回測績效好得不可思議」，不會拋任何例外。
只能靠測試抓。
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features.build import build, build_all, FEATURE_COLS, labels


def _panel():
    return pd.read_parquet(Path(__file__).resolve().parent.parent
                           / "data/features/panel.parquet")


def test_truncation_invariance():
    """把未來資料整段刪掉，as_of 當天的特徵值必須一模一樣。
    這直接證明：特徵沒有用到 as_of 之後的任何一列。"""
    p = _panel()
    dates = sorted(p["date"].unique())
    bad = []
    for d in dates[-40::8]:
        full = build(p, d).set_index("code")[FEATURE_COLS]
        truncated = build(p[p["date"] <= d], d).set_index("code")[FEATURE_COLS]
        diff = (full - truncated).abs().max().max()
        if pd.notna(diff) and diff > 1e-9:
            bad.append((str(d)[:10], float(diff)))
    assert not bad, f"特徵受未來資料影響：{bad}"


def test_fast_path_matches_safe_path():
    """快速路徑（build_all）必須與安全路徑（build）逐值相同。"""
    p = _panel()
    fast = build_all(p)
    dates = sorted(p["date"].unique())
    bad = []
    for d in dates[-30::10]:
        a = build(p, d).set_index("code")[FEATURE_COLS].sort_index()
        b = (fast[fast["as_of"] == d].set_index("code")[FEATURE_COLS].sort_index())
        diff = (a - b).abs().max().max()
        if pd.notna(diff) and diff > 1e-9:
            bad.append((str(d)[:10], float(diff)))
    assert not bad, f"快慢路徑不一致：{bad}"


def test_labels_are_strictly_future():
    """標籤必須落在 as_of 之後，且步數正好等於 horizon。"""
    p = _panel()
    dates = sorted(p["date"].unique())
    d = dates[-30]
    lab = labels(p, d, 5)
    assert len(lab) > 0
    assert (lab["target_date"] > d).all(), "標籤日期沒有嚴格落在未來"
    sub = p[p["code"] == lab.iloc[0]["code"]].sort_values("date").reset_index(drop=True)
    i = sub.index[sub["date"] == d][0]
    assert sub.loc[i + 5, "date"] == lab.iloc[0]["target_date"], "horizon 步數不符"


def test_feature_hash_is_stable():
    from features.build import feature_hash
    p = _panel()
    f = build(p, p["date"].max())
    r = f.iloc[0]
    assert feature_hash(r) == feature_hash(r), "特徵指紋不穩定"
    assert len(feature_hash(r)) == 16


def test_labels_all_matches_labels():
    """向量化標籤與逐日標籤必須完全一致。"""
    p = _panel()
    from features.build import labels_all
    dates = sorted(p["date"].unique())
    fast = labels_all(p, 5)
    # 取樣日期必須離資料尾端夠遠，否則標籤本來就還沒實現（空表 ≠ 不一致）
    for d in dates[-120:-20:25]:
        a = labels(p, d, 5).set_index("code")[["fwd_ret", "y"]].sort_index()
        b = (fast[fast["as_of"] == d].set_index("code")[["fwd_ret", "y"]].sort_index())
        assert len(a) == len(b), f"{d} 筆數不符 {len(a)} vs {len(b)}"
        assert (a - b).abs().max().max() < 1e-9, f"{d} 標籤不一致"


def test_training_set_excludes_unrealised_labels():
    """訓練集不得包含標籤尚未實現的樣本（最容易犯的偷看未來）。"""
    import pandas as pd
    from models.statistical import _training_set
    p = _panel()
    as_of = sorted(p["date"].unique())[-1]
    for h in (5, 20):
        X, y_dir, y_ret = _training_set(p, as_of, h)
        assert len(X) > 0, f"h={h} 訓練集為空"
        assert len(X) == len(y_dir) == len(y_ret), f"h={h} 特徵與標籤長度不符"
        from features.build import build_all, labels_all
        labs = labels_all(p[p["date"] <= as_of], h)
        dates = sorted(p[p["date"] <= as_of]["date"].unique())
        cutoff = dates[-(h + 1)]
        assert labs[labs["as_of"] <= cutoff]["target_date"].max() <= pd.Timestamp(as_of), \
            f"h={h} 訓練標籤落在 as_of 之後"


def test_features_are_finite():
    """特徵不得含 inf —— sklearn 會拋錯，而且 inf 比缺值更危險。"""
    import numpy as np
    from features.build import build_all
    p = _panel()
    f = build_all(p)
    bad = {c: int(np.isinf(f[c].to_numpy(dtype="float64")).sum())
           for c in FEATURE_COLS
           if np.isinf(f[c].to_numpy(dtype="float64")).any()}
    assert not bad, f"特徵含 inf：{bad}"


def test_no_impossible_daily_moves():
    """還原後不得有超過漲跌停（±10%）太多的單日變動。

    門檻設 30%：台股漲跌停 ±10%，但新上市／興櫃轉上市股確實會有 11~30% 的
    真實波動，把它們「修正」掉是製造假資料。>30% 則幾乎必然是分割或減資。
    若這條測試失敗，代表還原漏了某個公司行動，而它會讓動能特徵與標籤
    同時中毒 —— 這是靜默的、不會拋錯的致命污染。
    """
    p = _panel()
    p = p.sort_values(["code", "date"])
    p["chg"] = p.groupby("code")["close"].pct_change()
    bad = p[p["chg"].abs() > 0.30]
    detail = [(r.code, str(r.date.date()), f"{r.chg:+.1%}") for r in bad.itertuples()]
    assert not detail, f"仍有不可能的單日變動（公司行動還原不完整）：{detail[:10]}"


def test_models_emit_return_distribution():
    """每個模型都必須給出期望報酬與區間，否則無法做幅度加權比較。

    只預測方向會系統性誤導：P(up)=0.60 但上漲 +1%、下跌 -3% 的標的，
    期望報酬是 -0.6%，方向準確率漂亮卻賠錢。
    """
    import pandas as pd
    from features.build import build
    from models import baselines, statistical
    p = _panel()
    d = p["date"].max()
    f = build(p, d)
    required = {"prob_up", "exp_ret", "ret_q10", "ret_q90", "direction"}
    for name, fn in {**baselines.ALL, **statistical.ALL}.items():
        out = fn(f, 5, d, p)
        assert not out.empty, f"{name} 未出手"
        assert required <= set(out.columns), f"{name} 缺欄位：{required - set(out.columns)}"
        assert (out["ret_q10"] <= out["ret_q90"]).all(), f"{name} 分位數顛倒"
        # 方向必須與期望報酬一致，否則排序與下注邏輯會自相矛盾
        if name != "always_up":
            assert ((out["exp_ret"] >= 0) == (out["direction"] == 1)).all(), \
                f"{name} 方向與期望報酬不一致"
