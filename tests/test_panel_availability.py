"""panel 欄位的可用日（point-in-time）。

tests/test_no_lookahead.py 守的是「特徵計算有沒有讀到 panel 的未來列」；
panel 自己帶的可用日錯誤在它的視野之外 —— 2026-09-24 審核實測：把 rev_yoy 提前到期末、
把融資／法人整欄換成 t+1 的值，那組測試全部照樣通過。這一組補這一層。

這種外洩「不會讓回測失敗、只會讓回測變漂亮」，所以要在資料層守。
讀的是真 panel（在 .gitignore 內），CI 沒有就跳過 —— 但每天 daily.py prepare 會跑到它。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data/features/panel.parquet"


@pytest.fixture(scope="module")
def tw():
    if not PANEL.exists():
        pytest.skip("本機沒有 panel.parquet")
    p = pd.read_parquet(PANEL)
    p = p[p["market"] == "TW"] if "market" in p.columns else p
    return p.sort_values(["code", "date"])


def test_monthly_revenue_only_changes_on_or_after_the_10th(tw):
    """月營收的可用日是次月 10 日（法定期限）。值改變的那一天若早於 10 日，就是提前可用。
    每檔第一次出現不算（那是 panel 起點，不是公告）。"""
    bad = []
    for code, g in tw.groupby("code"):
        r = g["rev_yoy"]
        first = r.first_valid_index()
        if first is None:
            continue
        chg = g.loc[(r != r.shift()) & r.notna() & (g.index != first), "date"]
        early = chg[chg.dt.day < 10]
        if len(early):
            bad.append((code, early.dt.strftime("%Y-%m-%d").tolist()[:3]))
    assert not bad, f"rev_yoy 在 10 日前就變動（提前可用＝前視）：{bad[:5]}"


def _shift_corr(tw: pd.DataFrame, col: str, transform) -> dict[int, float]:
    """欄位（經 transform）對 ret_1 在位移 −1／0／+1 的日橫斷面 Spearman 平均。"""
    d = tw.copy()
    d["ret_1"] = d.groupby("code")["close"].pct_change()
    d["x"] = transform(d)
    out = {}
    for k in (-1, 0, 1):
        d["r"] = d.groupby("code")["ret_1"].shift(-k)     # k=+1 → 明天的報酬
        cs = []
        for _, g in d[d["date"] >= d["date"].max() - pd.Timedelta(days=400)].groupby("date"):
            g = g.dropna(subset=["x", "r"])
            if len(g) >= 20 and g["x"].std() > 0:
                c = g["x"].corr(g["r"], method="spearman")
                if pd.notna(c):
                    cs.append(c)
        out[k] = float(np.mean(cs)) if cs else 0.0
    return out


def test_institutional_flow_is_aligned_to_the_same_day(tw):
    """法人買賣超與當日報酬同日相關最強（買超推升當日價）。
    若最強的是 +1（明天的報酬），代表欄位被往前錯了一天 —— 那是外洩。"""
    if "foreign" not in tw.columns:
        pytest.skip("panel 沒有 foreign 欄")
    c = _shift_corr(tw, "foreign", lambda d: d["foreign"] / d.groupby("code")["volume"]
                    .transform(lambda s: s.rolling(20).mean()))
    assert abs(c[0]) > abs(c[1]) and abs(c[0]) > abs(c[-1]), f"法人買賣超對位異常：{c}"


def test_margin_balance_is_aligned_to_the_same_day(tw):
    """融資餘額變化同理：同日相關最強（融資追價與當日報酬反向）。"""
    if "margin_bal" not in tw.columns or tw["margin_bal"].notna().mean() < 0.3:
        pytest.skip("panel 沒有足夠的融資餘額")
    c = _shift_corr(tw, "margin_bal", lambda d: d.groupby("code")["margin_bal"].pct_change())
    assert abs(c[0]) > abs(c[1]), f"融資餘額對位異常（明日相關 ≥ 當日）：{c}"
