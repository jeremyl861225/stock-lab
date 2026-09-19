"""橫斷面評估必須逐市場計算（src/crosssec.py）。

台美的交易日與漲跌互不相干。把兩地擠進同一天的橫斷面排名，
等於拿台股的跌幅去決定美股的名次 —— 那個排名沒有意義。

另外釘住一個 2026-09-19 真的發生過的錯：新加的 `market` 變數撞到迴圈內
既有的 `mkt`（＝當日市場平均報酬），於是 market 欄位被寫成 −0.0065 這種數字。
它不拋錯、欄位照樣存在，只是內容全錯 —— 正是這個專案最怕的那種失效。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import crosssec as C


def _frame(markets=("TW", "US"), n_days=12, n=25, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mk in markets:
        for d in range(n_days):
            for c in range(n):
                rows.append({"as_of": f"202609{d+1:02d}", "horizon": 20,
                             "model": "claude", "market": mk,
                             "prob_up": 0.5 + (c - n / 2) * 0.004,
                             "actual_return": (c - n / 2) * 0.004 + rng.normal(0, 0.02)})
    return pd.DataFrame(rows)


def test_market_column_holds_market_names_not_returns():
    r = C.evaluate(_frame())
    assert set(r["market"]) == {"TW", "US"}, f"market 欄內容不是市場名：{list(r['market'])}"
    for v in r["market"]:
        assert isinstance(v, str)


def test_markets_are_evaluated_separately():
    r = C.evaluate(_frame())
    assert len(r) == 2, "兩個市場被併成同一個橫斷面"
    # 單獨算一個市場，結果必須與併算時的那一列相同
    alone = C.evaluate(_frame(markets=("TW",)))
    a = r[r["market"] == "TW"].iloc[0]
    b = alone.iloc[0]
    for col in ("rank_ic", "spread", "top_excess", "days"):
        assert abs(float(a[col]) - float(b[col])) < 1e-9, f"{col} 受另一個市場影響"


def test_no_market_column_still_works():
    """舊格式的回測結果（沒有 market 欄）不得讓評估掛掉。"""
    r = C.evaluate(_frame(markets=("TW",)).drop(columns=["market"]))
    assert len(r) == 1
    assert r["market"].isna().all()
    assert isinstance(C.report(r), str)


def test_report_renders_both_shapes():
    assert "TW" in C.report(C.evaluate(_frame()))
    assert isinstance(C.report(C.evaluate(_frame(markets=("TW",)).drop(columns=["market"]))), str)
