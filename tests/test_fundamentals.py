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
    """毛利率的合理上界是 ~105%，不是 100%。

    100% 以上確實會出現，但只在存貨跌價回升等會計迴轉的情況，幅度很小。
    實測 6446（藥華藥）2019Q1 營收 0.36 億、毛利 0.36 億 → 101.65%，
    那是真的。超過 105% 才代表對齊或單位壞了。

    營益率刻意不設下界：臨床階段生技公司營收 0.24 億、研發費用 3.55 億，
    營益率 −1488% 是事實不是錯誤。設下界會把真實資料誤判成 bug。
    """
    f = F.build()
    gm = f["gross_margin"].dropna()
    if len(gm):
        assert gm.max() <= 1.05, f"毛利率最大 {gm.max():.2f} 超過 105%"
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


def test_as_of_row_is_a_single_quarter():
    """as_of() 回傳的每一列必須全部來自同一季（且是最後一季有資料的）。

    起因：原本用 groupby().last()，而 pandas 的 .last() 是逐欄取最後一個
    非空值 —— 它會把不同季的數字拼成同一列。實測 1303（南亞）被拼成
    「2026Q2 的毛利率 18.9% ＋ 2025Q2 的 EPS 年增 −143%」，
    而它的 EPS TTM 其實是 −0.41 → +6.20 強勁轉正，方向完全相反。
    照拼出來的數字會寫出「毛利率改善但 EPS 大幅衰退」這種與事實相反的判斷。
    """
    d = pd.Timestamp("2026-09-17")
    snap = F.as_of(d)
    if snap.empty:
        return
    full = F.build()
    for _, row in snap.iterrows():
        g = full[(full["code"] == row["code"]) & (full["avail_date"] <= d)]
        if g.empty:
            continue
        g = g[g[F.FUND_COLS].notna().any(axis=1)]   # 整列作廢的季別會被跳過
        if g.empty:
            continue
        last = g.sort_values("date").iloc[-1]
        assert row["date"] == last["date"], (
            f"{row['code']} 的 as_of 列是 {row['date']}，但最新已公告季是 {last['date']}")
        for c in F.FUND_COLS:
            a, b = row[c], last[c]
            same = (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(a - b) < 1e-9)
            assert same, f"{row['code']} 的 {c} 來自別季（{a} vs {b}）"
