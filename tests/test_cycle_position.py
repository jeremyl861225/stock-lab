"""一年期規則的週期位置修正（models/rule_1y.py 1.1.0 起）。

存在的理由：上一版全部是橫斷面百分位，對「這是景氣循環的哪個位置」沒有辨識力。
2026-09-19 的實例：華邦電毛利率 66.2%（自身歷史長期 20–35%）、EPS 年增 +1232%，
因而拿到橫斷面第 98／第 100 百分位、被排到台股一年期最前段（P漲 0.621）；
而同一天手寫的美光論點講的正是這組數字的反面（P漲 0.48）。
兩檔同屬一個記憶體循環卻被判到兩端，差別只在覆蓋率不在判斷。

這一組守的是：週期高點不得成為任何形式的優勢，且已實查論點永不被自動改。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models import rule_1y as R
from features import fundamentals as F
from features import us_fundamentals as UF

ROOT = Path(__file__).resolve().parent.parent


def _frame(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "code": [f"{1000+i}" for i in range(n)],
        "gm_chg_4q": rng.normal(0.02, 0.03, n),
        "roe_ttm": rng.normal(0.15, 0.06, n),
        "eps_yoy": rng.normal(0.3, 0.5, n),
        "rev_cagr_3y": rng.normal(0.1, 0.08, n),
        "PER": rng.uniform(8, 60, n),
        "gm_self_pct": np.linspace(0.0, 1.0, n),
    })


def test_being_at_your_own_peak_is_never_an_advantage():
    """第一版寫成 mult = 1 − 1.5×cyc，讓『位於自身高點但橫斷面成長排名偏低』
    的公司拿到加分（負貢獻乘上負數變正）—— 那是反的。"""
    d = _frame()
    p_with = R.prob_up(d)[0]
    d0 = d.copy(); d0["gm_self_pct"] = np.nan          # 無週期資訊＝不修正
    p_without = R.prob_up(d0)[0]
    delta = (p_with - p_without).to_numpy()
    at_peak = d["gm_self_pct"].to_numpy() >= 0.95
    assert (delta[at_peak] <= 1e-9).all(), \
        f"位於自身毛利率高點卻被加分：{delta[at_peak]}"
    assert (delta <= 1e-9).all(), "週期位置修正只能扣分或不動"


def test_correction_is_monotone_in_cycle_position():
    """自身百分位越高，扣得越多。"""
    d = _frame()
    p = R.prob_up(d)[0]
    d0 = d.copy(); d0["gm_self_pct"] = np.nan
    delta = (p - R.prob_up(d0)[0]).to_numpy()
    hi = delta[d["gm_self_pct"].to_numpy() >= 0.95].mean()
    mid = delta[(d["gm_self_pct"].to_numpy() > 0.75)
                & (d["gm_self_pct"].to_numpy() < 0.95)].mean()
    lo = delta[d["gm_self_pct"].to_numpy() <= 0.75]
    assert (np.abs(lo) < 1e-9).all(), "75 百分位以下不該被修正"
    assert hi < mid < 1e-9, f"修正沒有隨週期位置遞增：hi={hi} mid={mid}"


def test_only_cyclical_signals_are_damped():
    """roe_ttm 與 rev_cagr_3y 是水準與三年趨勢，不隨單季循環位置調整。"""
    assert set(R.CYCLICAL) == {"gm_chg_4q", "eps_yoy", "PER"}
    for c in ("roe_ttm", "rev_cagr_3y"):
        assert c not in R.CYCLICAL


def test_missing_history_means_no_correction():
    """季數不足（美股目前）時 gm_self_pct 是 NaN，修正必須自動不生效 ——
    用 5 季的歷史講『相對自身歷史的極值』沒有意義。"""
    d = _frame()
    d["gm_self_pct"] = np.nan
    a = R.prob_up(d)[0]
    b = R.prob_up(d.drop(columns=["gm_self_pct"]))[0]
    assert (a - b).abs().max() < 1e-12


def test_gm_self_pct_is_point_in_time():
    """自身百分位只能用該季（含）之前的歷史算。

    用截斷不變性驗，不用手寫公式比對 —— 公式比對會被 NaN 的分母約定絆倒
    （expanding 的視窗把 NaN 算進分母），而那不是我們要守的性質。
    要守的是：把後面的季別整段刪掉，前面那些季的值必須一模一樣。
    """
    f = F.build()
    if f.empty or "gm_self_pct" not in f.columns:
        pytest.skip("尚無台股基本面")
    code = f.groupby("code")["gm_self_pct"].count().idxmax()
    g = f[f["code"] == code].sort_values("date").reset_index(drop=True)
    if g["gm_self_pct"].notna().sum() < 6:
        pytest.skip("季數不足")

    def recompute(series: pd.Series) -> pd.Series:
        return series.expanding(min_periods=F.SELF_PCT_MIN_Q).apply(
            lambda w: float((w <= w.iloc[-1]).mean()), raw=False)

    full = recompute(g["gross_margin"])
    assert (full - g["gm_self_pct"]).abs().max() < 1e-9, \
        "落盤的欄位與重算結果不同 —— 中間有別的步驟改過它"
    for cut in (len(g) - 1, len(g) - 3, len(g) - 5):
        if cut < F.SELF_PCT_MIN_Q:
            continue
        trunc = recompute(g["gross_margin"].iloc[:cut + 1])
        a, b = full.iloc[cut], trunc.iloc[cut]
        if pd.isna(a) and pd.isna(b):
            continue
        assert abs(float(a) - float(b)) < 1e-12, \
            f"第 {cut} 季的自身百分位受未來季別影響（{a} vs {b}）"


def test_both_markets_define_it_the_same_way():
    """名字一樣，定義就必須一樣，否則橫斷面比較會靜默失真。"""
    assert "gm_self_pct" in F.FUND_COLS
    assert "gm_self_pct" in UF.FUND_COLS
    assert F.SELF_PCT_MIN_Q == UF.SELF_PCT_MIN_Q


def test_derived_column_stays_out_of_the_basis_hash():
    """gm_self_pct 是 gross_margin 歷史的確定性函數，進雜湊只會在新增欄位
    那天讓全部標的失配、被誤記成『財報已更新』（METHOD §5.4 警告過的事）。"""
    import roll_1y as RL
    assert "gm_self_pct" in RL.HASH_EXCLUDE


def test_memory_cycle_pair_is_no_longer_split_across_the_book():
    """華邦電與南亞科在記憶體循環高點，修正後不該還排在最前段。"""
    f = F.as_of("2026-09-18")
    if f.empty or "gm_self_pct" not in f.columns:
        pytest.skip("尚無台股基本面")
    b = ROOT / "data/briefing.parquet"
    if not b.exists():
        pytest.skip("尚無簡報")
    per = pd.read_parquet(b)
    f = f.copy()
    f["PER"] = f["code"].map(dict(zip(per["code"], per["PER"])))
    f = f[f["gm_chg_4q"].notna() | f["eps_yoy"].notna()]
    p = R.prob_up(f)[0]
    f = f.assign(p=p.values)
    for code in ("2344", "2408"):
        row = f[f["code"] == code]
        if row.empty:
            continue
        rank = int((f["p"] > float(row["p"].iloc[0])).sum()) + 1
        assert rank > 3, f"{code} 仍排在第 {rank} 名 —— 週期位置修正沒有生效"
