"""一年期每日滾動的不變式。

這個模組存在的理由，就是不讓「每天更新」變成「每天製造雜訊」。
h=250 的日頻預測重疊 249/250，若 P漲 每天都動，看起來像持續學習，
實際上是把百分位排名的擾動包裝成判斷。以下測試守的就是這條界線。
"""
import json, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import roll_1y as R

PANEL = Path(__file__).resolve().parent.parent / "data/features/panel.parquet"
pytestmark = pytest.mark.skipif(
    not PANEL.exists() or not list((Path(__file__).resolve().parent.parent
                                    / "judgments").glob("*_1y*.json")),
    reason="需要面板與一年期判斷檔")


@pytest.fixture
def clean_state(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "STATE", tmp_path / "state.json")
    return R.STATE


def test_dry_run_does_not_touch_state(clean_state):
    """乾跑不得寫狀態檔。狀態檔記的是『上次真正發出的預測依據什麼』，
    乾跑若也寫，一次中止的執行就會把基準線往前推，
    下次真正滾動時全部標的都會被誤判成『輸入未變』。"""
    R.roll("2026-09-09")
    assert not clean_state.exists()


def test_unchanged_basis_reprices_only(clean_state):
    """輸入沒變的日子，P漲 必須一模一樣，只有價格相關的數字會動。"""
    a = R.roll("2026-09-09", commit=True)
    b = R.roll("2026-09-09")
    s = b["_summary"]
    assert s["repriced_only"] == s["total"], "輸入未變卻判定成重新判斷"
    pa = {x["code"]: x["prob_up"] for x in a["judgments"]}
    for x in b["judgments"]:
        assert x["prob_up"] == pa[x["code"]], f"{x['code']} 的 P漲 無故變動"
        assert x["repriced_only"] is True


def test_researched_thesis_is_never_auto_changed(clean_state):
    """已實查的論點是讀法說會與產業資料寫出來的。
    程式沒有那些資訊，硬套財報規則等於把有根據的判斷換成沒根據的。"""
    a = R.roll("2026-09-09", commit=True)
    b = R.roll("2026-09-10")          # 跨過 8 月營收公告日，基礎必變
    assert b["_summary"]["basis_changed"], "跨過月營收公告日卻沒偵測到基礎更新"
    pa = {x["code"]: x["prob_up"] for x in a["judgments"]}
    flagged = {x["code"] for x in b["_summary"]["needs_revision"]}
    for x in b["judgments"]:
        if x.get("researched"):
            assert x["prob_up"] == pa[x["code"]], f"{x['code']} 已實查卻被自動改了 P漲"
            assert x["code"] in flagged, f"{x['code']} 基礎更新卻沒標記待複核"


def test_as_of_cannot_exceed_panel(clean_state):
    """as_of 一律是資料日。允許滾到沒有資料的日期，
    會產生一個對不到任何收盤的預測，日後拿什麼價格結算就說不清楚。"""
    with pytest.raises(RuntimeError, match="超過面板最後一個交易日"):
        R.roll("2099-01-01")


def test_every_judgment_carries_provenance(clean_state):
    """每一筆滾動出來的預測都要能分辨『新判斷』與『同一判斷換了價格』。
    少了這個，250 筆重疊預測看起來就像 250 次獨立下注。"""
    r = R.roll("2026-09-09", commit=True)
    for x in r["judgments"]:
        for k in ("thesis_as_of", "repriced_only", "basis", "roll_note",
                  "close_at_call"):
            assert k in x, f"{x['code']} 缺少 {k}"
        assert x["ret_q10"] <= x["exp_ret"] <= x["ret_q90"]
