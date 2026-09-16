"""基本面特徵的資料品管測試。"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import fundamentals as F


def test_pub_date_uses_legal_deadline():
    """公告日必須用法定期限，不能用季末日 —— 用季末日就是前視偏差。"""
    assert F._pub_date(pd.Timestamp("2026-06-30")) == pd.Timestamp("2026-08-14")
    assert F._pub_date(pd.Timestamp("2026-03-31")) == pd.Timestamp("2026-05-15")
    assert F._pub_date(pd.Timestamp("2025-12-31")) == pd.Timestamp("2026-03-31")


def test_as_of_never_returns_unpublished():
    """as_of(D) 回傳的每一列，其 avail_date 都必須 <= D。"""
    for d in ("2025-06-01", "2026-01-15", "2026-09-17"):
        f = F.as_of(d)
        if f.empty:
            continue
        assert (f["avail_date"] <= pd.Timestamp(d)).all(), f"{d} 取到了尚未公告的財報"


def test_margins_within_possible_range():
    """毛利率不可能 >100% 或 <-200%；超出代表資料或對齊壞了。"""
    f = F.build()
    gm = f["gross_margin"].dropna()
    if len(gm):
        assert gm.max() <= 1.0, f"毛利率最大 {gm.max():.2f} 超過 100%"
        assert gm.min() >= -2.0, f"毛利率最小 {gm.min():.2f} 低於 -200%"


def test_roe_not_from_income_statement():
    """ROE 若誤取損益表的同名欄位會算出 200–600%。有值時必須落在合理範圍。"""
    f = F.build()
    roe = f["roe_ttm"].dropna()
    if len(roe) > 20:
        assert roe.quantile(0.99) < 1.5, f"ROE 99 分位 {roe.quantile(0.99):.2f}，疑似取到損益表欄位"


def test_eps_yoy_guards_tiny_base():
    """基期接近零時不給 eps_yoy，否則會出現 +1400% 這種假成長。"""
    f = F.build()
    if "eps_ttm" in f.columns:
        base = f.groupby("code")["eps_ttm"].shift(4)
        assert f.loc[base.abs() < 0.5, "eps_yoy"].notna().sum() == 0
