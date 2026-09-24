"""成績單的數學要用已知答案驗算（src/score.py、src/accuracy.py）。

2026-09-24 審核抓到三個「數字存在但定義錯」的問題，現有 36 個測試全部靜默通過：
  1. 中性判斷被記成看多進命中率；
  2. p 值用 i.i.d. 二項檢定，同日 50 檔＋重疊下假陽性 0.16–0.44；
  3. Brier 技能對當批事後上漲比率算，任何常數預測必為負。
這一組守的是：棄權不進分母、天數不足不出 p、技能對事前基本率、
以及 prob_up 判方向的準確率與 correct 是兩個數。
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import score as S
import accuracy as A


def _rows(n_days=1, n_codes=10, model_prob=0.7, model_dir=1, neutral=0):
    rows = []
    for d in range(n_days):
        for i in range(n_codes):
            up = i % 2 == 0                                # 一半上漲
            actual = 1 if up else -1
            for m, prob, pdir in (("always_up", 0.55, 1), ("m", model_prob, model_dir)):
                p, e, dr = prob, 0.02 * pdir, pdir
                if m == "m" and i < neutral:
                    p, e, dr = 0.5, 0.0, 0                 # 中性列
                rows.append({"pid": f"{m}{d}{i}", "settled_at_utc": "x", "model": m,
                             "model_family": "f", "model_version": "1", "horizon": 5,
                             "code": f"{1000+i}", "as_of": f"2026090{d+1}",
                             "target_date": "20260930", "price_start": 1, "price_end": 1,
                             "actual_return": 0.03 if up else -0.03,
                             "actual_direction": actual, "predicted_direction": dr,
                             "prob_up": p, "correct": None if dr == 0 else int(dr == actual),
                             "brier": (p - (1 if up else 0)) ** 2, "exp_ret": e,
                             "in_interval": 1, "interval_score": 0.1, "market": "TW"})
    return rows


def _wire(tmp_path, monkeypatch, rows):
    f = tmp_path / "s.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(S, "SETTLEMENTS", f)
    monkeypatch.setattr(A, "SETTLEMENTS", f)
    monkeypatch.setattr(A, "PREDICTIONS", tmp_path / "none.jsonl")
    return f


def test_abstain_rows_leave_the_hit_rate_denominator(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, _rows(n_codes=10, neutral=4))
    m = {x["model"]: x for x in S.summary()["by_horizon"]["5"]["models"]}["m"]
    assert m["n"] == 10 and m["n_abstain"] == 4 and m["n_scored"] == 6
    # 非中性的 6 檔（i=4..9）：看多，上漲的是偶數 → 4,6,8 命中 = 3/6
    assert abs(m["accuracy"] - 0.5) < 1e-9
    # 舊定義會把 4 筆中性當看多：i=0,2 命中 → 5/10；新定義不受影響
    assert m["brier"] > 0                                  # 中性列的 Brier 0.25 照計


def test_abstain_is_derived_from_immutable_fields_even_if_correct_was_written(tmp_path, monkeypatch):
    """已結算的舊列（append-only）寫了 correct=1，讀取端仍要認出它是中性。"""
    rows = _rows(n_codes=4, neutral=2)
    for r in rows:
        if r["model"] == "m" and r["predicted_direction"] == 0:
            r["predicted_direction"], r["correct"] = 1, 1     # 模擬 2026-09-23 前的舊列
    _wire(tmp_path, monkeypatch, rows)
    m = {x["model"]: x for x in S.summary()["by_horizon"]["5"]["models"]}["m"]
    assert m["n_abstain"] == 2 and m["n_scored"] == 2


def test_direction_by_prob_is_reported_separately(tmp_path, monkeypatch):
    """correct 量的是 sign(exp_ret) 的決策，accuracy_by_prob 量 prob_up>0.5。
    momentum 型的模型（exp_ret 被歷史均值主導）兩者可以差很多。"""
    _wire(tmp_path, monkeypatch, _rows(model_prob=0.3, model_dir=1))   # 機率看空、期望值看多
    m = {x["model"]: x for x in S.summary()["by_horizon"]["5"]["models"]}["m"]
    assert abs(m["accuracy"] - 0.5) < 1e-9                  # 看多：一半命中
    assert abs(m["accuracy_by_prob"] - 0.5) < 1e-9          # 看空：另一半命中
    assert abs(m["inconsistency_rate"] - 1.0) < 1e-9


def test_no_p_value_when_effective_days_are_too_few(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, _rows(n_days=3))          # 3 天 ÷ 重疊 5 = 0.6 有效天
    m = {x["model"]: x for x in S.summary()["by_horizon"]["5"]["models"]}["m"]
    assert m["p_value_vs_base"] is None and m["edge_t_adj"] is None
    assert m["n_days"] == 3 and m["n_eff_days"] < S.MIN_EFF_DAYS


def test_cluster_t_folds_by_overlap():
    d = pd.Series([0.1] * 30 + [0.2] * 30)                 # 60 天的逐日配對差
    t1, n1 = S.cluster_t(d, overlap=1.0)
    t5, n5 = S.cluster_t(d, overlap=5.0)
    assert n1 == 60 and n5 == 12
    assert abs(t5 - t1 / np.sqrt(5)) < 1e-9


def test_binomial_p_matches_normal_approximation():
    # 獨立樣本下的舊檢定仍要對；只是成績單不再用它
    assert abs(S._binom_p_iid(60, 100, 0.5) - 0.0228) < 0.002


def test_brier_skill_uses_longrun_base_rate(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, _rows())
    blk = S.summary()["by_horizon"]["5"]
    lr = S.LONGRUN_BASE_RATE[("TW", 5)]
    au = {x["model"]: x for x in blk["models"]}["always_up"]
    assert abs(au["brier_skill"] - (1 - au["brier"] / (lr * (1 - lr)))) < 1e-3
    # 舊定義：當批基本率 0.5 → 常數 0.55 的技能必為負
    assert au["brier_skill_insample"] < 0


def test_accuracy_summary_excludes_abstain_and_reports_layers(tmp_path, monkeypatch):
    _wire(tmp_path, monkeypatch, _rows(n_codes=10, neutral=4))
    a = A.summary(model="m")
    assert a["settled"] == 10 and a["abstain"] == 4 and a["scored"] == 6
    assert abs(a["hit_rate"] - 0.5) < 1e-9
    h5 = a["by_horizon"]["5"]
    assert h5["mz"]["slope"] is None                       # 單日不出斜率
    assert "rank_ic" in h5["mz"] and "mean_actual" in h5
