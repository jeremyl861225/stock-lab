# -*- coding: utf-8 -*-
"""SEC companyfacts 解析的不變式。

這五條各自對應一個**已經實際發生過**的失真。全部用合成 fact，不連網 ——
測試不該因為 SEC 當天慢就變紅，也不該因為某家公司改了申報方式就變紅。
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from collect import sec_edgar as SE


def _fact(start, end, val, filed, form="10-Q", accn="x"):
    d = {"end": end, "val": val, "filed": filed, "form": form, "accn": accn}
    if start:
        d["start"] = start
    return d


def _facts(**tags):
    """{標籤: [fact, ...]} → companyfacts 的 facts 結構。"""
    return {k: {"units": {"USD": v}} for k, v in tags.items()}


def test_non_periodic_forms_are_ignored():
    """委託書裡的數字不算「首次公布」。

    實測 CAT 的 FY2021 淨利在 NetIncomeLoss 底下只有一筆，來源是 2026-04-30
    的 DEF 14A（薪酬對照表要揭露前五年淨利）。照單全收的話，2021Q4 的
    可用日會變成四年後 —— 那一整季在回測裡憑空消失。
    """
    f = _facts(NetIncomeLoss=[
        _fact("2021-01-01", "2021-12-31", 6_489, "2026-04-30", form="DEF 14A"),
        _fact("2021-01-01", "2021-12-31", 6_489, "2022-02-16", form="10-K"),
    ])
    got, _tags = SE._collect(f, ["NetIncomeLoss"], instant=False)
    assert got[("2021-01-01", "2021-12-31")]["filed"] == "2022-02-16"

    only_proxy = _facts(NetIncomeLoss=[
        _fact("2021-01-01", "2021-12-31", 6_489, "2026-04-30", form="DEF 14A")])
    got2, _ = SE._collect(only_proxy, ["NetIncomeLoss"], instant=False)
    assert got2 == {}, "只有委託書有的期別不該被採用"


def test_as_originally_reported_beats_restatement():
    """同一期間被重報時取最早申報的那一版。

    用重編後的數字去填當時的位置是前視 —— 重編本身就是後來才發生的事。
    """
    f = _facts(GrossProfit=[
        _fact("2025-01-01", "2025-03-31", 100, "2025-04-30", form="10-Q"),
        _fact("2025-01-01", "2025-03-31", 111, "2026-04-30", form="10-K"),
    ])
    got, _ = SE._collect(f, ["GrossProfit"], instant=False)
    assert got[("2025-01-01", "2025-03-31")]["val"] == 100


def test_tags_splice_only_when_they_agree():
    """兩個標籤在重疊期間對不上時不准接起來。

    實測 AXP：2017 起的 RevenueFromContractWithCustomer 對金融股只算合約收入
    （單季約 60 億），2016 以前的 Revenues 是總收入淨額（約 80 億）。
    補洞式合併會讓 2017Q1 的 rev_yoy 憑空出現 −25%，公司什麼都沒發生。
    """
    disagree = _facts(
        Revenues=[_fact("2016-01-01", "2016-03-31", 80, "2016-04-27"),
                  _fact("2016-04-01", "2016-06-30", 82, "2016-07-26"),
                  _fact("2017-01-01", "2017-03-31", 79, "2017-04-27")],
        RevenueFromContractWithCustomerExcludingAssessedTax=[
                  _fact("2017-01-01", "2017-03-31", 60, "2017-04-27"),
                  _fact("2017-04-01", "2017-06-30", 62, "2017-07-25")],
    )
    got, tags = SE._collect(
        disagree, ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"],
        instant=False)
    assert tags == ["Revenues"], f"對不上的標籤被接了進來：{tags}"
    assert ("2017-04-01", "2017-06-30") not in got, "接了一段換過定義的歷史"

    # 重疊期間一致時就該接起來，用來延長歷史
    agree = _facts(
        SalesRevenueNet=[_fact("2015-01-01", "2015-03-31", 50, "2015-04-27"),
                         _fact("2016-01-01", "2016-03-31", 55, "2016-04-27")],
        Revenues=[_fact("2016-01-01", "2016-03-31", 55, "2016-04-27"),
                  _fact("2017-01-01", "2017-03-31", 60, "2017-04-27"),
                  _fact("2018-01-01", "2018-03-31", 65, "2018-04-27")],
    )
    got2, tags2 = SE._collect(agree, ["Revenues", "SalesRevenueNet"], instant=False)
    assert set(tags2) == {"Revenues", "SalesRevenueNet"}
    assert got2[("2015-01-01", "2015-03-31")]["val"] == 50, "一致的標籤沒有接起來"


def test_cumulative_cashflow_is_differenced_into_quarters():
    """現金流量表是年初至今累計，必須還原成單季。

    實測 AAPL 的 NetCashProvidedByUsedInOperatingActivities 期間長度是
    3／6／9／12 個月。照單季用的話，第四季的營運現金流會變成整年的數字，
    資本支出強度跟著錯 —— 台股那邊就是這樣算出 76% 的。
    """
    cum = {("2025-01-01", "2025-03-31"): {"val": 10.0, "filed": "2025-04-30"},
           ("2025-01-01", "2025-06-30"): {"val": 25.0, "filed": "2025-07-30"},
           ("2025-01-01", "2025-09-30"): {"val": 45.0, "filed": "2025-10-30"},
           ("2025-01-01", "2025-12-31"): {"val": 60.0, "filed": "2026-02-10"}}
    q = SE._quarters_from_cumulative(cum)
    assert [round(q[k]["val"], 6) for k in sorted(q)] == [10.0, 15.0, 20.0, 15.0]
    assert q["2025-12-31"]["filed"] == "2026-02-10"


def test_fiscal_year_end_quarter_is_annual_minus_three():
    """財年最後一季沒有第四份 10-Q，必須用年報減同財年前三季。"""
    dur = {("2025-01-01", "2025-03-31"): {"val": 10.0, "filed": "2025-04-30"},
           ("2025-04-01", "2025-06-30"): {"val": 12.0, "filed": "2025-07-30"},
           ("2025-07-01", "2025-09-30"): {"val": 11.0, "filed": "2025-10-30"},
           ("2025-01-01", "2025-12-31"): {"val": 47.0, "filed": "2026-02-10"}}
    q = SE._quarters_from_duration(dur)
    assert round(q["2025-12-31"]["val"], 6) == 14.0
    assert q["2025-12-31"]["derived"] is True
    # 可用日必須是年報那天，不是第三季那天 —— 少一天都算前視
    assert q["2025-12-31"]["filed"] == "2026-02-10"

    # 前三季少一季就不准推。少一筆會把兩季的金額算成一季。
    del dur[("2025-04-01", "2025-06-30")]
    assert "2025-12-31" not in SE._quarters_from_duration(dur)


def test_same_period_end_two_start_dates_takes_earliest_filed():
    """同一期末有多個起始日時，取最早申報的那一筆，而且每次都一樣。

    實測 GS 的 2011 財年營運現金流有兩筆：
      (2010-12-31 → 2011-12-31) 365 天　216.45 億　filed 2012-02-28  當年的 10-K
      (2011-01-01 → 2011-12-31) 364 天　225.01 億　filed 2014-02-28  兩年後重編
    原本靠 set 走訪順序決定取哪一筆 —— 而 Python 的字串雜湊每個行程都不同，
    於是同一份輸入每次跑出不同答案，GS 的檔案每天都進 commit。
    這一條同時守「取原始申報值」與「可重現」。
    """
    dur = {("2010-12-31", "2011-12-31"): {"val": 21_645, "filed": "2012-02-28"},
           ("2011-01-01", "2011-12-31"): {"val": 22_501, "filed": "2014-02-28"}}
    assert SE._fiscal_years(dur) == [("2010-12-31", "2011-12-31")]

    # 單季同理
    q = {("2011-09-30", "2011-12-31"): {"val": 5.0, "filed": "2012-02-28"},
         ("2011-10-01", "2011-12-31"): {"val": 6.0, "filed": "2014-02-28"}}
    got = SE._quarters_from_duration(q)
    assert got["2011-12-31"]["val"] == 5.0


def test_span_classifies_52_53_week_fiscal_quarters():
    """財報季不是 90 天。52／53 週制下單季會落在 84~98 天，
    分類區間太窄會讓整家公司的季別憑空消失。"""
    assert SE._span("2026-03-29", "2026-06-27") == 1      # AAPL 實際季別
    assert SE._span("2025-12-28", "2026-06-27") == 2
    assert SE._span("2025-09-29", "2026-06-27") == 3
    assert SE._span("2025-06-30", "2026-06-27") == 4
    assert SE._span("2026-06-01", "2026-06-27") is None   # 對不上任何一格


@pytest.mark.skipif(not (ROOT / "data/raw/sec_fund").exists(),
                    reason="需要 data/raw/sec_fund")
def test_no_quarter_claims_to_predate_its_own_period_end():
    """申報日不可能早於期末日。出現就是解析把期間對錯了。"""
    import json
    bad = []
    for f in sorted((ROOT / "data/raw/sec_fund").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for e, q in d["quarters"].items():
            if q["filed_first"] <= e:
                bad.append((d["code"], e, q["filed_first"]))
    assert not bad, f"申報日早於期末日：{bad[:5]}"
