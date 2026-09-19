# -*- coding: utf-8 -*-
"""美股 point-in-time 成分股的不變式。

這些守的是「回測會變漂亮、但不會報錯」那一類 —— 與台股 PIT 同一個目的，
只是美股的還原方式不同（自己用申報股數重算市值，不是抄 ETF 籃子）。
"""
import json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from collect import pit_universe_us as PU
from collect import us as US


def test_split_between_filings_is_applied():
    """分割發生在兩次申報之間時，股數必須補乘分割倍數。

    NVDA 2024-06-10 分割 10:1，而上一次申報股數是 2024-05-29 的 24.6 億
    （基準日 2024-05-24，分割前），下一次要等 2024-08-28 的 245.3 億。
    中間兩個半月若直接用 24.6 億乘分割後的股價，市值會少算十倍 ——
    NVDA 會整個掉出前 50，而那正是這段期間漲最多的一檔。
    """
    rows = [{"end": "2024-05-24", "shares": 2.46e9, "filed": "2024-05-29"}]
    splits = {"2024-06-10": 10.0}
    assert PU.pit_shares(rows, splits, "2024-06-05") == pytest.approx(2.46e9)
    assert PU.pit_shares(rows, splits, "2024-06-11") == pytest.approx(24.6e9)
    assert PU.pit_shares(rows, splits, "2024-08-01") == pytest.approx(24.6e9)


def test_split_before_the_share_count_is_not_double_counted():
    """分割早於股數基準日時不能再乘一次 —— 那筆申報本來就是分割後的數字。"""
    rows = [{"end": "2024-08-23", "shares": 24.53e9, "filed": "2024-08-28"}]
    assert PU.pit_shares(rows, {"2024-06-10": 10.0}, "2024-09-15") \
        == pytest.approx(24.53e9)


def test_unfiled_share_count_is_invisible():
    """還沒申報的股數不能用。用了就是未來函數，而且不會報錯。"""
    rows = [{"end": "2024-05-24", "shares": 2.46e9, "filed": "2024-05-29"},
            {"end": "2024-08-23", "shares": 24.53e9, "filed": "2024-08-28"}]
    assert PU.pit_shares(rows, {}, "2024-08-27") == pytest.approx(2.46e9)
    assert PU.pit_shares(rows, {}, "2024-08-28") == pytest.approx(24.53e9)
    assert PU.pit_shares(rows, {}, "2024-01-01") is None


@pytest.mark.skipif(not (ROOT / "config/universe_us").exists(),
                    reason="需要 config/universe_us")
def test_us_load_respects_as_of():
    """`us.load(as_of)` 必須回傳不晚於 as_of 的那一份。

    這個參數原本是假的：簽名收下 as_of 卻永遠回傳最新的 universe，
    於是呼叫端以為自己做了 point-in-time，實際上拿的是今天的成分股。
    台股那支一直是真的 —— 兩邊介面一樣、行為不同，最難發現。
    """
    files = sorted((ROOT / "config/universe_us").glob("*.json"))
    assert files, "沒有任何美股快照"
    stamps = [f.stem for f in files]
    got = US.load(stamps[-1])
    assert got["as_of"] <= stamps[-1]

    if len(stamps) >= 2:
        mid = stamps[len(stamps) // 2]
        assert US.load(mid)["as_of"] <= mid
        # 早於第一份快照的日期必須拋錯，不可以拿最新的頂替
        with pytest.raises(ValueError):
            US.load("19900101")


@pytest.mark.skipif(not (ROOT / "config/universe_us").exists(),
                    reason="需要 config/universe_us")
def test_snapshots_declare_their_method():
    """每份快照都要說得出自己是怎麼來的。

    美股的快照有兩種來源：`collect/us.py` 的即時市值（只有最近幾天），
    與本模組還原的歷史市值。兩者是同一個定義、不同的輸入，
    但品質不同 —— 混在同一個目錄裡而不標記，日後沒有人分得出來。
    """
    for f in sorted((ROOT / "config/universe_us").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        assert d.get("method"), f"{f.name} 沒有 method 欄位"
        assert d.get("market") == "US"


def test_share_scale_converts_to_quote_units():
    """股數要換算成報價單位才算得出市值。

    兩種會錯、而且都不會報錯的情形：
      存託憑證　TSM 申報的是台灣普通股，一股 ADS ＝ 5 股 → 市值算成 4.76 倍
      多重股權　BRK-B 沒有維度的那筆只有 A 股 94 萬股 → 市值算成 4.7 億
                而不是 1.09 兆，這檔兆元公司於是安靜地掉出前 50
    """
    rows = [{"end": "2026-06-30", "shares": 1_000_000, "filed": "2026-07-30"}]
    assert PU.pit_shares(rows, {}, "2026-09-15") == pytest.approx(1_000_000)
    assert PU.pit_shares(rows, {}, "2026-09-15", 1500.0) == pytest.approx(1.5e9)
    assert PU.pit_shares(rows, {}, "2026-09-15", 0.21) == pytest.approx(210_000)


def test_reconcile_checks_both_directions():
    """對帳要查兩側：重建算錯的，和重建漏掉的。

    只查前者的話，一檔被算成十分之一市值的公司會直接掉出榜外，
    而對帳從頭到尾不會提到它 —— 錯得越嚴重越不會被發現。
    """
    live = {"AAPL": 4.0e12, "BRK-B": 1.09e12, "MSFT": 3.0e12}
    cons = [{"code": "AAPL", "mcap": 4.02e12}, {"code": "MSFT", "mcap": 2.98e12}]
    bad = PU.recon_diff(cons, live)
    assert any("BRK-B" in b for b in bad), "漏掉的成分股沒有被叫出來"
    assert not any("AAPL" in b for b in bad)

    # 榜上但算錯的也要抓
    cons2 = [{"code": "AAPL", "mcap": 0.4e12}, {"code": "BRK-B", "mcap": 1.1e12},
             {"code": "MSFT", "mcap": 3.0e12}]
    assert any("AAPL" in b for b in PU.recon_diff(cons2, live))
