"""外洩守門（src/leak_canary.py）。

守的是三件事：
  1. guard() 對「知道答案」的模型會冒煙，對誠實的模型不會；
  2. 已存的天花板要在合理範圍（猜漲 < 市場神諭 < 相對強弱 < 100%）；
  3. 既有回測結果（若在本機）對照天花板不得冒煙 —— 任何模型的方向準確率
     一旦進到 77–86% 以上，第一個該懷疑的是外洩，不是方法有效。
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import leak_canary as L

ROOT = Path(__file__).resolve().parent.parent
ORC = [{"market": "TW", "horizon": 5, "always_up": 0.54, "perfect_market_timing": 0.63,
        "perfect_relative_strength_mean": 0.80, "perfect_relative_strength_median": 0.86,
        "perfect_with_magnitude": 1.0}]


def _results(acc_target: float, leak: bool, n_days=30, n=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for i in range(n):
            ret = rng.normal(0, 0.03)
            y = 1 if ret > 0 else -1
            if leak:
                prob = 0.5 + 0.4 * np.sign(ret)           # 知道答案
                dirn = y
            else:
                prob = float(np.clip(rng.normal(0.55, 0.05), 0.05, 0.95))
                dirn = 1 if rng.random() < acc_target else -y   # 控制準確率
            rows.append({"as_of": f"{d:04d}", "horizon": 5, "market": "TW", "pit": True,
                         "model": "leaky" if leak else "honest", "code": str(i),
                         "prob_up": prob, "direction": dirn, "actual_direction": y,
                         "actual_return": ret, "correct": int(dirn == y),
                         "brier": (prob - (y > 0)) ** 2})
    return pd.DataFrame(rows)


def test_guard_flags_a_model_that_knows_the_answer():
    flags = L.guard(_results(0.5, leak=True), ORC)
    assert flags and flags[0]["model"] == "leaky"


def test_guard_is_quiet_for_an_honest_model():
    assert L.guard(_results(0.55, leak=False), ORC) == []


def test_saved_ceilings_are_ordered_sensibly():
    p = ROOT / "data/backtest/leak_canary.json"
    if not p.exists():
        pytest.skip("尚未產生 leak_canary.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    for r in d["oracles"]:
        assert r["always_up"] < r["perfect_market_timing"] < r["perfect_relative_strength_mean"] <= 1.0
        assert r["perfect_relative_strength_median"] >= r["perfect_relative_strength_mean"] - 0.02
        # 誠實天花板本身就低於 90%：這一行是「>90% 只能來自外洩」的量化依據
        assert r["perfect_market_timing"] < 0.75
    for c in d.get("canaries", []):
        assert c["leak_fwd_ret"] > 0.95, "把答案當特徵應接近 100%"
        assert c["honest"] < 0.75, "誠實模型不可能到 75%"


def test_existing_backtest_does_not_trip_the_smoke_detector():
    rp, cp = ROOT / "data/backtest/results.parquet", ROOT / "data/backtest/leak_canary.json"
    if not (rp.exists() and cp.exists()):
        pytest.skip("本機沒有回測結果或天花板")
    res = pd.read_parquet(rp)
    orc = json.loads(cp.read_text(encoding="utf-8"))["oracles"]
    flags = L.guard(res, orc)
    assert not flags, f"回測結果冒煙 —— 先查外洩再談準確率：{flags}"
