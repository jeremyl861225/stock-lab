"""事件日曆（src/events.py）。

2026-09-19 補上的缺口：系統原本完全沒有事件日曆，而「財報是否落在視窗內」
是 5／20 日最大的遺漏條件變數 —— 它同時決定實現波動與漂移。
這一組守兩件事：事件算得對，以及**它絕不進統計模型的特徵集**。
"""
import json, sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import events as EV
from features.build import FEATURE_COLS

ROOT = Path(__file__).resolve().parent.parent


def test_event_columns_never_enter_the_model_feature_set():
    """未來的財報日是『今天查得到』而非『as_of 查得到』的資訊。
    放進訓練集就是前視偏差，而它不會拋錯、只會讓回測變漂亮。"""
    for c in EV.EVENT_COLS + ["evt_types_20"]:
        assert c not in FEATURE_COLS, f"{c} 混進了 FEATURE_COLS —— 這是前視偏差"


def test_only_future_events_are_reported():
    as_of = pd.Timestamp("2026-09-18")
    e = EV.upcoming(["2330", "1101"], ["TW", "TW"], as_of)
    for d in e["evt_date"].dropna():
        assert pd.Timestamp(d) > as_of, f"回報了已經發生的事件 {d}"
    assert (e["evt_days"].dropna() > 0).all()


def test_taiwan_monthly_revenue_is_the_tenth():
    """台股月營收法定於次月 10 日前公告 —— 9/18 的下一個事件就是 10/10。"""
    e = EV.upcoming(["1101"], ["TW"], pd.Timestamp("2026-09-18"))
    r = e.iloc[0]
    assert r["evt_type"] == "月營收"
    assert r["evt_date"] == "2026-10-10"
    assert r["evt_in_20"] and not r["evt_in_5"]


def test_window_flags_are_consistent():
    e = EV.upcoming(["1101", "2330"], ["TW", "TW"], pd.Timestamp("2026-09-18"))
    for _, r in e.iterrows():
        if pd.isna(r["evt_days"]):
            assert not r["evt_in_5"] and not r["evt_in_20"]
            continue
        assert r["evt_in_5"] == (r["evt_days"] <= 5)
        assert r["evt_in_20"] == (r["evt_days"] <= 20)
        if r["evt_in_5"]:
            assert r["evt_in_20"], "5 日內的事件必然也在 20 日內"


def test_missing_us_cache_degrades_to_no_event_not_a_crash():
    """沒抓過排程的美股標的要回報『無事件』，不能讓整份簡報掛掉。"""
    e = EV.upcoming(["___NOSUCH___"], ["US"], pd.Timestamp("2026-09-18"))
    assert len(e) == 1
    assert e.iloc[0]["evt_date"] is None
    assert not e.iloc[0]["evt_in_20"]


def test_briefing_carries_the_event_columns():
    b = ROOT / "data/briefing.parquet"
    if not b.exists():
        pytest.skip("尚無簡報")
    d = pd.read_parquet(b)
    for c in ("evt_type", "evt_date", "evt_days", "evt_in_5", "evt_in_20"):
        assert c in d.columns, f"簡報缺少 {c} —— 判斷時就看不到事件"
    assert d["evt_in_20"].fillna(False).any(), "所有標的都沒有事件，資料多半沒接上"


def test_adr_gives_taiwan_issuers_their_real_earnings_date():
    """2330 與 TSM 是同一家公司、同一場法說會。
    台股沒有免費的法說會日期來源，但 ADR 那邊有確切日期，該用上。

    2026-09-18 的實例：TSM 財報 2026-10-15 落在 20 日視窗內 ——
    那正是當天 2330 的判斷裡人工寫下的那句話。"""
    if not (ROOT / "data/raw/us_events/TSM.json").exists():
        pytest.skip("尚未抓過 TSM 的財報排程")
    e = EV.upcoming(["2330"], ["TW"], pd.Timestamp("2026-09-18")).iloc[0]
    assert "財報" in (e["evt_types_20"] or ""), \
        f"2330 的 20 日視窗內沒有帶到 ADR 的財報日：{e['evt_types_20']}"


def test_verifier_flags_unmentioned_earnings_but_not_routine_revenue():
    """視窗內有財報卻沒提到 → 要警告；但月營收是台股每檔每月都有的例行事件，
    逐檔警告等於 50 項噪音（WEEKLY.md：噪音太多的警報等於沒有警報）。"""
    import verify_judgment as V
    assert "月營收" not in V.EVENT_WORTH_FLAGGING
    assert set(V.EVENT_WORTH_FLAGGING) == {"財報", "除權息"}

    b = pd.DataFrame({
        "code": ["AAA", "BBB", "CCC"],
        "evt_in_20": [True, True, True],
        "evt_in_5": [False, False, False],
        "evt_types_20": ["財報", "財報", "月營收"],
        "evt_date": ["2026-10-13", "2026-10-13", "2026-10-10"],
        "evt_days": [17.0, 17.0, 16.0],
    })
    js = [{"horizon": 20, "judgments": [
        {"code": "AAA", "thesis": "估值偏高", "inference": "", "rationale": ""},
        {"code": "BBB", "thesis": "10/13 財報是關鍵", "inference": "", "rationale": ""},
        {"code": "CCC", "thesis": "估值偏高", "inference": "", "rationale": ""},
    ]}]
    out = V.check_event_window(b, js)
    flagged = {x.split()[1] for x in out}
    assert flagged == {"AAA"}, f"應只警告 AAA，實際 {flagged}"
