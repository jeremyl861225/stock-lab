"""mech_core 基準線：手寫判斷的機械核心。

存在的理由（2026-09-19 實測）：2026-09-18 台股 50 檔的手寫 p20 與
「20 日反轉 × 營收年增」的 Spearman 是 +0.736，與 momentum 基準線是 −0.62。
判斷有三分之二可被這兩句話複製，而且與最便宜的既有基準線剛好是鏡像 ——
兩個鏡像模型裡必有一個看起來很準，那不是證據。
所以要證明判斷有增量價值，它必須贏過這條線，不是贏過猜漲。

這一組守的是這條線本身的性質：PIT、分市場、不因缺資料而偷改權重。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features.build import build, FEATURE_COLS
from models import baselines

PANEL = Path(__file__).resolve().parent.parent / "data/features/panel.parquet"
SAMPLE = Path(__file__).resolve().parent / "fixtures" / "panel_sample.parquet"


def _panel():
    return pd.read_parquet(PANEL if PANEL.exists() else SAMPLE)


def test_mech_core_is_registered_and_never_abstains():
    """基準線每天必須出手，沒有例外（憲法 D）。"""
    assert "mech_core" in baselines.ALL
    p = _panel()
    d = p["date"].max()
    f = build(p, d)
    for h in (5, 20):
        out = baselines.mech_core(f, h, d, p)
        assert len(out) == len(f), f"h={h} 有標的沒出手"
        assert {"prob_up", "exp_ret", "ret_q10", "ret_q90", "direction"} <= set(out.columns)
        assert out["prob_up"].between(0.30, 0.70).all()
        assert (out["ret_q10"] <= out["ret_q90"]).all()
        assert (out["ret_q10"] > -1.0).all(), "隱含負股價"
        assert ((out["exp_ret"] >= 0) == (out["direction"] == 1)).all()


def test_direction_is_reversal_not_momentum():
    """跌深的要拿到比較高的機率 —— 否則它就只是 momentum 換個名字。"""
    p = _panel()
    d = p["date"].max()
    f = build(p, d)
    out = baselines.mech_core(f, 20, d, p).merge(
        f[["code", "ret_20"] + (["market"] if "market" in f else [])], on="code")
    g = out[out["market"] == "TW"] if "market" in out else out
    g = g.dropna(subset=["ret_20"])
    c = g["prob_up"].corr(g["ret_20"], method="spearman")
    assert c < -0.2, f"與 20 日報酬的相關是 {c:+.2f}，方向不是反轉"


def test_base_rate_is_per_market():
    """共用一個基本率等於拿台股的多頭去墊高美股。"""
    p = _panel()
    if "market" not in p.columns or p["market"].nunique() < 2:
        pytest.skip("樣本面板只有一個市場")
    d = p["date"].max()
    f = build(p, d)
    out = baselines.mech_core(f, 20, d, p).merge(f[["code", "market"]], on="code")
    rates = out.groupby("market")["rationale"].first()
    assert rates.nunique() > 1 or True       # 文字可能相同；真正要比的是數字
    tw = baselines._hist_up_rate(p[p["market"] == "TW"], d, 20)
    us = baselines._hist_up_rate(p[p["market"] == "US"], d, 20)
    assert abs(tw - us) > 1e-9, "測試前提不成立：兩市場基本率恰好相同"
    for mk, want in (("TW", tw), ("US", us)):
        g = out[out["market"] == mk]
        assert f"{want:.1%}" in g["rationale"].iloc[0], \
            f"{mk} 用的不是自己的基本率：{g['rationale'].iloc[0]}"


def test_missing_revenue_market_does_not_silently_halve_the_tilt():
    """整個市場都沒有月營收時，權重要退回純反轉，
    不能讓 fillna(0) 把營收那一半變成中性分而偷偷把反轉的權重折半。"""
    p = _panel()
    d = p["date"].max()
    f = build(p, d).copy()
    if "market" not in f.columns:
        pytest.skip("需要 market 欄位")
    f.loc[f["market"] == "US", "rev_yoy"] = np.nan
    out = baselines.mech_core(f, 20, d, p).merge(f[["code", "market"]], on="code")
    us = out[out["market"] == "US"]
    f_us = f[f["market"] == "US"].copy()
    f_us["rev_yoy"] = np.nan
    alone = baselines.mech_core(f_us, 20, d, p[p["market"] == "US"])
    a = us.set_index("code")["prob_up"].sort_index()
    b = alone.set_index("code")["prob_up"].sort_index()
    spread_ratio = (a - a.mean()).std() / (b - b.mean()).std()
    assert abs(spread_ratio - 1.0) < 1e-6, \
        f"無月營收市場的傾斜幅度被改變了（比值 {spread_ratio:.3f}）"


def test_no_lookahead_in_mech_core():
    """截斷未來資料後，同一天的輸出必須一模一樣。"""
    p = _panel()
    dates = sorted(p["date"].unique())
    d = dates[-25]
    a = baselines.mech_core(build(p, d), 20, d, p[p["date"] <= d])
    b = baselines.mech_core(build(p[p["date"] <= d], d), 20, d, p[p["date"] <= d])
    x = a.set_index("code")[["prob_up", "exp_ret"]].sort_index()
    y = b.set_index("code")[["prob_up", "exp_ret"]].sort_index()
    assert (x - y).abs().max().max() < 1e-12, "mech_core 受未來資料影響"
