"""檢查點的月營收可用日（src/checkpoints.py::_monthly_rev）。

2026-09-21 抓到的靜默錯誤：FinMind 月營收表的 `date` 欄**已經是營收月的
次月**（8 月營收那一列 date=2026-09-01、revenue_month=8）。原本的程式在
它上面又加了一次 MonthBegin(1)，於是可用日比法定公告日晚整整一個月 ——
每一條 rev_yoy／rev_yoy_ttm 檢查點都拿上上個月的營收在對帳。

不會拋錯、不會有 NaN，只會讓論點的前提用過期的數字評分：
當天實際造成 3037／3008／1326 三檔誤報「動搖」，而
「亮起來的時候一定有事」正是這個機制的全部價值。

features/panel.py 用 revenue_year／revenue_month 自己重算，是對的。
同一件事有兩套定義時，這組測試釘住的是「兩套要一致」。
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import checkpoints as CP
from config import RAW


def _raw_rows():
    d = RAW / "finmind" / "rev"
    rows = []
    for f in sorted(d.glob("*.json")):
        rows += json.loads(f.read_text(encoding="utf-8")).get("payload", [])
    return rows


def test_finmind_date_column_is_the_month_after_the_revenue_month():
    """整條修正建立在這個前提上，所以先把前提本身釘住。

    FinMind 哪天改了 `date` 的語意，這一條會先紅 —— 那時 avail_date
    的算法要跟著改，而不是等檢查點默默對錯月份的帳。
    """
    rows = _raw_rows()
    if not rows:
        pytest.skip("本機沒有 FinMind 月營收快取")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    expect = df["revenue_year"].astype(int) * 12 + df["revenue_month"].astype(int) + 1
    actual = df["date"].dt.year * 12 + df["date"].dt.month
    bad = int((expect != actual).sum())
    assert len(df) >= 1000, f"樣本只有 {len(df)} 列，這個斷言等於沒跑"
    assert bad == 0, f"{bad}/{len(df)} 列的 date 不等於營收月+1，語意變了"


def test_availability_is_the_tenth_of_the_month_after_the_revenue_month():
    """法定公告期限：營收月的次月 10 日。8 月營收 → 9/10 可用。"""
    mr = CP._monthly_rev()
    if mr.empty:
        pytest.skip("本機沒有 FinMind 月營收快取")
    checked = 0
    for _, r in mr.iterrows():
        # date 已是營收月的次月，故法定公告日就是它當月的 10 日
        want = pd.Timestamp(r["date"]).replace(day=10)
        assert pd.Timestamp(r["avail_date"]) == want, (
            f"{r['code']} {r['date']}：可用日 {r['avail_date']}，應為 {want}")
        checked += 1
    assert checked >= 1000, f"只比對了 {checked} 列，這個測試在空轉"


def test_checkpoints_and_panel_agree_on_the_latest_available_month():
    """checkpoints 與 features/panel 對「今天看得到哪個月」必須一致。

    兩邊各自實作同一個概念，就是這次出錯的根因。這一條讓分歧當場變紅。
    """
    mr = CP._monthly_rev()
    pnl_p = ROOT / "data/features/panel.parquet"
    if mr.empty or not pnl_p.exists():
        pytest.skip("本機沒有月營收或面板")
    pnl = pd.read_parquet(pnl_p, columns=["date", "code", "market", "rev_yoy"])
    pnl = pnl[pnl["market"] == "TW"]
    as_of = pnl["date"].max()
    latest_panel = (pnl[pnl["date"] == as_of].dropna(subset=["rev_yoy"])
                    .set_index("code")["rev_yoy"])
    seen = mr[mr["avail_date"] <= as_of].sort_values("avail_date")
    latest_cp = seen.groupby("code")["rev_yoy"].last()
    compared = 0
    for code, v in latest_panel.items():
        if code not in latest_cp.index or pd.isna(latest_cp[code]):
            continue
        assert latest_cp[code] == pytest.approx(float(v), abs=1e-6), (
            f"{code}：panel 看到 {v:.4f}，checkpoints 看到 "
            f"{latest_cp[code]:.4f} —— 兩邊對到不同的月份")
        compared += 1
    assert compared >= 20, f"只比對了 {compared} 檔，這個測試在空轉"
