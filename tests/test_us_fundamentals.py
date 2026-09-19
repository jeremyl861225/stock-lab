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


def test_avail_basis_is_declared_and_consistent(f):
    """可用日必須說得出自己是哪來的，而且與那個來源一致。

    2026-09-19 起可用日有兩種來源：SEC 的真實申報日，以及取不到時退回的
    法定上限。這一條擋的是「退回發生了但沒人知道」—— 退回本身沒問題，
    靜默退回才有問題（`eps_yoy_basis` 是同一個道理）。
    """
    assert set(f["avail_basis"]) <= {"SEC申報日", "法定上限"}

    legal_rows = f[f["avail_basis"] == "法定上限"]
    lag = (legal_rows["avail_date"] - legal_rows["date"]).dt.days.unique()
    allowed = {UF.LAG_Q, UF.LAG_FY, UF.LAG_Q_FPI, UF.LAG_FY_FPI}
    assert set(lag) <= allowed, f"退回法定上限卻不是法定天數：{sorted(set(lag) - allowed)}"

    sec_rows = f[f["avail_basis"] == "SEC申報日"]
    sec_lag = (sec_rows["avail_date"] - sec_rows["date"]).dt.days
    assert (sec_lag <= UF.MAX_FILED_LAG).all(), "採信了離期末過遠的申報日"


def test_sec_avail_date_equals_first_filing(f):
    """標成 SEC申報日 的列，日期必須真的等於該季首次申報日。

    中間只要有一個地方把它換成推估值（例如順手 fillna 成法定上限），
    整套就退回舊行為而外觀不變 —— 這正是這個 repo 最常見的失效方式。
    """
    checked = 0
    for code, path in UF._sources():
        raw = json.loads(path.read_text(encoding="utf-8"))
        qs = raw.get("quarters") or {}
        if not qs or "filed_first" not in next(iter(qs.values())):
            continue
        rows = f[(f["code"] == code) & (f["avail_basis"] == "SEC申報日")]
        for _, r in rows.iterrows():
            q = qs.get(str(r["date"].date()))
            assert q is not None, f"{code} {r['date']} 在原始檔找不到"
            assert r["avail_date"] == pd.Timestamp(q["filed_first"]), \
                f"{code} {r['date']} 可用日不等於首次申報日"
            checked += 1
    assert checked > 0, "沒有任何一列走 SEC 申報日，這個測試等於沒跑"


def test_negative_equity_voids_roe(f):
    """**平均**股東權益為負時 ROE 必須作廢。

    ABBV 的權益一度是 −59 億美元（長年大額買回），照算會得到 −206%，
    而它其實穩定獲利 —— 負分母會把訊號的正負號整個翻過來，
    在橫斷面百分位裡直接把最賺錢的一批排到最後。

    判準是**平均**（(當期 + 四季前) / 2），因為那才是 roe_ttm 的分母。
    這裡原本寫成「單季權益為負」—— 在 yfinance 只給 5–7 季的時候兩者
    剛好等價，換成 SEC 的 18 年歷史之後就不等價了：ABBV 2018Q2 的權益
    由 +60 億翻成 −34 億，平均仍是 +13 億，分母沒有翻號，ROE 該算。
    對著實作沒有主張的東西下斷言，遲早會擋掉正確的改動。
    """
    for _code, p in UF._sources():
        q, _ = UF._load_one(p)
        if q.empty or "equity" not in q.columns:
            continue
        code = q["code"].iloc[0]
        q = q.sort_values("date")
        eq = q["equity"].astype(float)
        avg = (eq + eq.shift(4)) / 2
        neg = q[avg <= 0]
        if neg.empty:
            continue
        rows = f[(f["code"] == code) & f["date"].isin(neg["date"])]
        assert rows["roe_ttm"].isna().all(), f"{code} 平均權益為負卻算出了 ROE"


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
    for _code, p in UF._sources():
        q, a = UF._load_one(p)
        if q.empty or a.empty or "revenue" not in q.columns or "revenue" not in a.columns:
            continue
        a = a.dropna(subset=["revenue"]).sort_values("date")
        for _, ar in a.iterrows():
            # 2012 年之前不檢查。XBRL 是 2009–2011 分階段強制的，更早的期別
            # 只以「後來財報的比較欄」進入 companyfacts，季與年常常來自
            # 不同次重編（AAPL FY2009 的遞延收入重編、GE 2009 的停業部門）。
            # 那不是這支程式的錯，而回測期間也從來用不到 2012 以前。
            if ar["date"] < pd.Timestamp("2012-01-01"):
                continue
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
