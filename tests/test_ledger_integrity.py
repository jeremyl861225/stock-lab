"""帳本完整性：這些測試守的是這套系統的憲法。"""
import json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import PREDICTIONS
from predict import _pid


def _rows():
    if not PREDICTIONS.exists():
        return []
    return [json.loads(l) for l in PREDICTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_pid_is_pure_function_of_slot():
    """pid 不得含模型版本。含版本等於替同一天同一檔重開一個槽，
    改模型重跑就變成下兩次注，只要有一次對就進帳。"""
    a = _pid("20260916", 20, "claude", "2330")
    b = _pid("20260916", 20, "claude", "2330", "9.9.9")
    assert a == b, "pid 受 model_version 影響 —— 同一個槽會被重開"


def test_revision_numbers_are_unique_per_pid():
    """同一個 pid 的修訂編號必須各不相同，否則無法分辨修訂順序。
    舊版把每次修訂都寫 revision=1，實測 20260916 美股 53 檔各有兩筆 rev=1。"""
    g = defaultdict(list)
    for r in _rows():
        if r.get("revision") is not None:
            g[r["pid"]].append(r["revision"])
    bad = {k: v for k, v in g.items() if len(v) != len(set(v))}
    # 歷史殘留容許存在，但不得再增加：只檢查 2026-09-17 之後寫入的列
    new = defaultdict(list)
    for r in _rows():
        if r.get("created_at_utc", "") >= "2026-09-17" and r.get("revision") is not None:
            new[r["pid"]].append(r["revision"])
    bad_new = {k: v for k, v in new.items() if len(v) != len(set(v))}
    assert not bad_new, f"新寫入的列有重複修訂編號：{list(bad_new)[:3]}"


def test_no_settled_prediction_is_mutated():
    """已結算的預測不得再被修改 —— 這是事後改答案。"""
    from config import SETTLEMENTS
    if not SETTLEMENTS.exists():
        return
    settled = {}
    for l in SETTLEMENTS.read_text(encoding="utf-8").splitlines():
        if l.strip():
            s = json.loads(l)
            settled[s["pid"]] = s.get("settled_at_utc", "")
    for r in _rows():
        t = settled.get(r["pid"])
        if t and r.get("created_at_utc", "") > t:
            raise AssertionError(f"pid {r['pid']} 在結算後仍被寫入（結算 {t}，寫入 {r['created_at_utc']}）")


def test_scoring_path_deduplicates():
    """結算與計分都必須對 (as_of, 期別, 模型, 代號) 去重取最新，
    否則歷史殘留的舊公式列會被當成獨立下注。"""
    src = (Path(__file__).resolve().parent.parent / "src")
    for f, needle in ((src / "settle.py", 'dedup['),
                      (src / "accuracy.py", 'drop_duplicates(')):
        assert needle in f.read_text(encoding="utf-8"), f"{f.name} 失去去重"
