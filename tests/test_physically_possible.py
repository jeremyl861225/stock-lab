"""物理可能性關卡。

存在的理由：這套系統曾把**負的股價**送上線 —— 面板的「保守價」欄位
顯示南電 −316.5 元、創意 −914.7 元，38 檔裡 18 檔的 q10 低於 −90%。
原因是用常態分佈算分位數（q10 = 期望值 − 1.36σ），而常態的左尾會
延伸到 −∞；一年期、年化波動 90% 的標的就會掉到 −96%。

當時有 36 個測試，沒有一個抓到。因為它們全在測「有沒有偷看未來」
「版面有沒有垮」—— 沒有一個在問「這個數字有沒有可能存在」。

這一組就是補那個缺口。判準只有一條：
**如果一個數字在現實中不可能出現，它就是 bug，不管公式看起來多合理。**
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PREDICTIONS, DATA

ROOT = Path(__file__).resolve().parent.parent


def _effective() -> pd.DataFrame:
    """帳本裡「結算與計分實際會用到」的那一筆。

    帳本是附加式的，修訂會留下舊列 —— 那是刻意的，不刪列是憲法。
    所以要套用與 settle.py／accuracy.py 相同的去重規則
    （(as_of, horizon, model, code) 取最新），否則會把已被取代的
    歷史錯誤當成現行問題。
    """
    if not PREDICTIONS.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in PREDICTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows).sort_values("created_at_utc")
    return d.drop_duplicates(["as_of", "horizon", "model", "code"], keep="last")


def test_no_prediction_implies_negative_price():
    """報酬率不可能低於 −100% —— 那代表股價變成負數。

    常態分佈的分位數在長期間高波動下會越過這條線。
    見 src/models/quantiles.py：改用對數常態。
    """
    d = _effective()
    if d.empty:
        return
    for col in ("ret_q10", "dn_magnitude", "exp_ret"):
        if col not in d.columns:
            continue
        v = d[col].dropna()
        bad = d.loc[v[v <= -1.0].index]
        assert bad.empty, (
            f"{len(bad)} 筆的 {col} ≤ −100%（隱含負股價）："
            f"{bad[['as_of','horizon','code',col]].head(3).to_dict('records')}")


def test_quantiles_are_ordered_and_bracket_expectation():
    """q10 ≤ 期望值 ≤ q90。順序反了或期望值落在區間外，
    代表區間與中心值不是同一個分佈算出來的。"""
    d = _effective()
    if d.empty or "ret_q10" not in d.columns:
        return
    q = d.dropna(subset=["ret_q10", "ret_q90", "exp_ret"])
    bad = q[(q.ret_q10 > q.ret_q90) | (q.ret_q10 > q.exp_ret) | (q.exp_ret > q.ret_q90)]
    assert bad.empty, f"{len(bad)} 筆的分位數順序錯誤或未包住期望值"


def test_probabilities_are_probabilities():
    d = _effective()
    if d.empty:
        return
    p = d["prob_up"].dropna()
    bad = p[(p <= 0) | (p >= 1)]
    assert bad.empty, f"{len(bad)} 筆的 prob_up 不在 (0,1)"


def test_panel_shows_no_negative_or_zero_price():
    """面板上任何一個價格都必須為正。這是最後一道 —— 前面全漏掉時，
    使用者看到的就是 NT$-316.5。"""
    import re
    f = ROOT / "docs/index.html"
    if not f.exists():
        return
    h = f.read_text(encoding="utf-8")
    neg = re.findall(r"<i>(?:NT\$|US\$)(-[\d,\.]+)</i>", h)
    zero = re.findall(r"<i>(?:NT\$|US\$)(0(?:\.0+)?)</i>", h)
    assert not neg, f"面板出現 {len(neg)} 個負價格：{neg[:5]}"
    assert not zero, f"面板出現 {len(zero)} 個零價格"


def test_fundamental_ratios_are_physically_possible():
    """財務比率的物理上限。超出代表近零分母或跨表對齊壞了 ——
    實測曾出現 ROE 610%（取到損益表的同名欄位）、
    存貨天數 34,197 天（零營收期的極小成本基數）。"""
    from features import fundamentals as F
    f = F.build()
    if f.empty:
        return
    for col, lo, hi, why in (
            ("gross_margin", -2.0, 1.05, "毛利率"),
            ("debt_ratio", 0.0, 1.0, "負債比"),
            ("roe_ttm", -2.0, 1.5, "ROE（>150% 通常是取到損益表欄位）"),
            ("capex_intensity", 0.0, 3.0, "資本支出強度"),
            ("inv_days_chg", -730.0, 730.0, "存貨天數變化（兩年為物理上限）")):
        v = f[col].dropna()
        if not len(v):
            continue
        bad = v[(v < lo) | (v > hi)]
        assert bad.empty, (
            f"{why} 有 {len(bad)} 筆超出 [{lo}, {hi}]，"
            f"範圍 {v.min():.2f}~{v.max():.2f}")


def test_lognormal_quantiles_never_cross_minus_one():
    """分位數函式本身的性質：不論波動多大，q10 永遠 > −100%。"""
    from models.quantiles import quantiles
    import numpy as np
    for vol_annual in (0.1, 0.5, 1.0, 2.0, 5.0):
        sd = vol_annual / np.sqrt(252) * np.sqrt(250)
        for ev in (-0.5, 0.0, 0.5, 2.0):
            q10, q90 = quantiles(ev, sd)
            assert q10 > -1.0, f"波動 {vol_annual}、期望 {ev} 時 q10={q10}"
            assert q10 <= q90, f"波動 {vol_annual} 時 q10 > q90"
