"""結算的算術要用已知答案驗算（src/settle.py）。

2026-09-24 審核：settle.py 是計分的唯一入口，卻沒有任何測試執行過它 ——
test_ledger_integrity 只 grep 原始碼裡有沒有 'dedup[' 這個字串。
去重改壞可以讓結算筆數灌水 3.8 倍、重複結算可以讓帳本翻倍，現有測試都抓不到。
這一組用 tmp_path 造一份 3 檔 × 幾天的 panel 與預測，逐筆對已知答案。
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import settle as S


def _panel(tmp_path):
    days = pd.bdate_range("2026-09-01", periods=12)
    rows = []
    for code, closes in (("1101", [10, 11, 12, 11, 13, 14, 13, 15, 16, 15, 17, 18]),
                         ("2330", [100, 99, 98, 100, 97, 96, 95, 94, 93, 92, 91, 90]),
                         ("AAPL", [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50])):
        for d, c in zip(days, closes):
            rows.append({"date": d, "code": code, "close": float(c),
                         "market": "US" if code == "AAPL" else "TW"})
    f = tmp_path / "features"; f.mkdir()
    pd.DataFrame(rows).to_parquet(f / "panel.parquet", index=False)
    return f, days


def _pred(pid, as_of, code, h, prob, exp, q10=-0.05, q90=0.05, created="2026-09-01T00:00:00"):
    return {"pid": pid, "as_of": as_of, "code": code, "horizon": h, "model": "m",
            "model_family": "x", "model_version": "1.0", "prob_up": prob, "exp_ret": exp,
            "ret_q10": q10, "ret_q90": q90, "direction": 1 if exp >= 0 else -1,
            "created_at_utc": created}


def _wire(tmp_path, monkeypatch, preds):
    feats, days = _panel(tmp_path)
    pp = tmp_path / "predictions.jsonl"
    pp.write_text("\n".join(json.dumps(p) for p in preds), encoding="utf-8")
    sp = tmp_path / "settlements.jsonl"
    monkeypatch.setattr(S, "PREDICTIONS", pp)
    monkeypatch.setattr(S, "SETTLEMENTS", sp)
    monkeypatch.setattr(S, "FEATURES", feats)
    return sp, days


def _rows(sp):
    return [json.loads(l) for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_return_direction_brier_and_interval_are_computed_from_known_prices(tmp_path, monkeypatch):
    d0 = "20260901"
    sp, days = _wire(tmp_path, monkeypatch, [
        _pred("a", d0, "1101", 5, 0.7, 0.03),          # 10 → 14：+40%，看多，命中
        _pred("b", d0, "2330", 5, 0.4, -0.02),         # 100 → 96：−4%，看空，命中
        _pred("c", d0, "1101", 20, 0.6, 0.02),         # 20 日後還沒到期 → pending
    ])
    out = S.run()
    assert out["settled"] == 2 and out["pending"] == 1
    r = {x["pid"]: x for x in _rows(sp)}
    a, b = r["a"], r["b"]
    assert abs(a["actual_return"] - 0.4) < 1e-6 and a["actual_direction"] == 1
    assert a["predicted_direction"] == 1 and a["correct"] == 1
    assert abs(a["brier"] - (0.7 - 1) ** 2) < 1e-6
    assert a["in_interval"] == 0                        # +40% 落在 q90=+5% 之外
    # interval score = 寬度 + 10×越界量
    assert abs(a["interval_score"] - (0.10 + 10 * (0.4 - 0.05))) < 1e-6
    assert a["target_date"] == days[5].strftime("%Y%m%d")
    assert abs(b["actual_return"] + 0.04) < 1e-6 and b["correct"] == 1
    assert abs(b["brier"] - (0.4 - 0) ** 2) < 1e-6 and b["in_interval"] == 1
    assert b["market"] == "TW"


def test_neutral_prediction_is_settled_as_abstain(tmp_path, monkeypatch):
    """p=0.5 且期望值 0 的列：方向 0、correct 空白，Brier 仍是 0.25。"""
    sp, _ = _wire(tmp_path, monkeypatch, [_pred("n", "20260901", "1101", 5, 0.5, 0.0)])
    S.run()
    r = _rows(sp)[0]
    assert r["predicted_direction"] == 0 and r["correct"] is None
    assert abs(r["brier"] - 0.25) < 1e-9


def test_only_latest_revision_is_settled_and_nothing_settles_twice(tmp_path, monkeypatch):
    d0 = "20260901"
    sp, _ = _wire(tmp_path, monkeypatch, [
        _pred("old", d0, "1101", 5, 0.3, -0.02, created="2026-09-01T00:00:00"),
        _pred("new", d0, "1101", 5, 0.8, +0.04, created="2026-09-01T01:00:00"),   # 修訂版
    ])
    out = S.run()
    assert out["settled"] == 1 and out["superseded_dropped"] == 1
    assert _rows(sp)[0]["pid"] == "new"
    again = S.run()                                     # 再跑一次：不得重複結算
    assert again["settled"] == 0 and len(_rows(sp)) == 1


def test_us_and_missing_codes(tmp_path, monkeypatch):
    sp, _ = _wire(tmp_path, monkeypatch, [
        _pred("u", "20260901", "AAPL", 5, 0.6, 0.01),   # 50 → 50：報酬 0，記為未上漲
        _pred("x", "20260901", "9999", 5, 0.6, 0.01),   # panel 裡沒有 → missing
    ])
    out = S.run()
    assert out["settled"] == 1 and out["missing"] == 1
    r = _rows(sp)[0]
    assert r["market"] == "US" and r["actual_direction"] == -1 and r["correct"] == 0
