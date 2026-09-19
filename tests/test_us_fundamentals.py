"""美股基本面特徵的不變式。

這些測試守的全是「數字看起來有根據、實際上是除法或缺值的產物」那一類錯誤 ——
它們不會讓程式失敗，只會讓判斷失真，所以必須用測試擋。
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import us_fundamentals as UF

pytestmark = pytest.mark.skipif(not (ROOT / "data/raw/us_fund").exists(),
                                reason="需要 data/raw/us_fund 快取")


@pytest.fixture(scope="module")
def f():
    return UF.build()


def test_avail_date_is_always_after_period_end(f):
    """可用日必須晚於季末日。相等或更早就是前視偏差 ——
    它不會讓回測失敗，只會讓回測變漂亮，所以特別危險。"""
    assert (f["avail_date"] > f["date"]).all()


def test_filing_lag_matches_legal_deadline(f):
    """申報落後天數只能是四個法定值之一。
    出現其他數字代表 _avail 被繞過，或年度／季度判斷錯了。"""
    lag = (f["avail_date"] - f["date"]).dt.days.unique()
    allowed = {UF.LAG_Q, UF.LAG_FY, UF.LAG_Q_FPI, UF.LAG_FY_FPI}
    assert set(lag) <= allowed, f"出現非法定落後天數：{sorted(set(lag) - allowed)}"


def test_negative_equity_voids_roe(f):
    """平均股東權益為負時 ROE 必須作廢。

    ABBV 的權益是 −59 億美元（長年大額買回），照算會得到 −206%，
    而它其實穩定獲利 —— 負分母會把訊號的正負號整個翻過來，
    在橫斷面百分位裡直接把最賺錢的一批排到最後。
    """
    for p in sorted((ROOT / "data/raw/us_fund").glob("*.json")):
        q, _ = UF._load_one(p)
        if q.empty or "equity" not in q.columns:
            continue
        code = q["code"].iloc[0]
        neg = q[q["equity"] < 0]
        if neg.empty:
            continue
        rows = f[(f["code"] == code) & f["date"].isin(neg["date"])]
        assert rows["roe_ttm"].isna().all(), f"{code} 權益為負卻算出了 ROE"


def test_ratios_stay_inside_physical_bounds(f):
    """比率不能落在物理上不可能的區間。分母趨近零時會算出
    毛利率 251%、資本支出強度 76% 這種數字（台股那邊實測過）。"""
    assert f["gross_margin"].dropna().between(-1, 1).all()
    assert f["op_margin"].dropna().between(-5, 2).all()
    assert f["capex_intensity"].dropna().between(0, 3).all()
    assert f["debt_ratio"].dropna().between(0, 1).all()
    assert f["rev_yoy"].dropna().between(-1, 20).all()


def test_quarterly_sums_reconcile_with_annual():
    """四季加總必須對得上年報（±3%）。

    這一條在擋「累計制」—— 台股的現金流量表是年初至今累計，
    照單季用會算出資本支出強度 76%。美股的季度報表是單季值，
    但那是上游（yfinance）的行為，不是保證；哪天它改成累計制，
    這個測試會在判斷被寫壞之前先失敗。
    """
    bad = []
    for p in sorted((ROOT / "data/raw/us_fund").glob("*.json")):
        q, a = UF._load_one(p)
        if q.empty or a.empty or "revenue" not in q.columns or "revenue" not in a.columns:
            continue
        a = a.dropna(subset=["revenue"]).sort_values("date")
        for _, ar in a.iterrows():
            w = q[(q["date"] > ar["date"] - pd.Timedelta(days=370))
                  & (q["date"] <= ar["date"])].dropna(subset=["revenue"])
            if len(w) != 4 or not float(ar["revenue"]):
                continue
            if abs(w["revenue"].sum() / float(ar["revenue"]) - 1) > 0.03:
                bad.append((q["code"].iloc[0], str(ar["date"].date())))
    assert not bad, f"四季加總對不上年報（疑似累計制）：{bad[:5]}"


def test_as_of_needs_revenue_not_just_any_column():
    """as_of 取的那一季必須有營收。

    實測 AMZN 2026Q2 在上游只有 EPS 一欄，其餘全空。用「任一欄不是空的」
    當判準會選中它，於是毛利率、ROE、資本支出強度全部變成缺值，
    而它 2026Q1 那一季其實是完整的。沒有營收就算不出任何比率。
    """
    s = UF.as_of("2026-09-18")
    assert not s.empty
    assert s["revenue"].notna().all()


def test_as_of_respects_point_in_time():
    """as_of 只能取已申報的季別。抓到未來的季報就是未來函數。"""
    d = pd.Timestamp("2026-06-30")
    s = UF.as_of(d)
    assert (s["avail_date"] <= d).all()
    assert not s.empty


def test_one_row_per_code():
    """每檔只能一列。用 groupby().last() 會逐欄取最後一個非空值，
    把不同季的數字拼成同一列（台股 1303 實測被拼成兩季混合）。"""
    s = UF.as_of("2026-09-18")
    assert s["code"].is_unique
