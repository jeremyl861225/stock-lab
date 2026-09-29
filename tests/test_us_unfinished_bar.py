# -*- coding: utf-8 -*-
"""紐約盤中抓到的今天那根 K 棒不可以進價量檔。

2026-09-29 台北 22:05 跑 prepare，yfinance 給的 9/29 Close 是即時價、不是 NaN，
結果盤中價被當收盤拿去結算與重新定價，而且完全沒有錯誤訊息。
"""
import datetime as dt, sys
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from collect import us as US

NY = ZoneInfo("America/New_York")
IDX = pd.DatetimeIndex(["2026-09-25", "2026-09-28", "2026-09-29"]).tz_localize(NY)
H = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=IDX)


def test_intraday_bar_is_dropped():
    out = US.drop_unfinished(H, dt.datetime(2026, 9, 29, 10, 5, tzinfo=NY))
    assert pd.Timestamp(out.index[-1]).date() == dt.date(2026, 9, 28)


def test_bar_kept_after_close():
    out = US.drop_unfinished(H, dt.datetime(2026, 9, 29, 17, 0, tzinfo=NY))
    assert len(out) == 3


def test_morning_before_open_keeps_yesterday():
    """台北早上 7:30 ＝ 紐約前一天晚上，前一天那根已收盤，不可誤砍。"""
    out = US.drop_unfinished(H, dt.datetime(2026, 9, 29, 19, 30, tzinfo=NY))
    assert len(out) == 3
    out = US.drop_unfinished(H.iloc[:2], dt.datetime(2026, 9, 29, 8, 0, tzinfo=NY))
    assert len(out) == 2
