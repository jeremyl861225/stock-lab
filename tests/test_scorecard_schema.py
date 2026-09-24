"""成績單的列必須是同一個 schema（src/score.py）。

2026-09-19 差點出事：新增「同名多版本另外分列」時，分版本的列少了
`edge_vs_always_up`，而 `report.py` 直接讀那個鍵 —— 一旦有結算資料
且某個模型有兩個版本，整份報表就會 KeyError。
而那要等到 2026-09-23 第一批結算才會炸，測試當下什麼都看不出來。

這一組守的是：不管哪一種列，鍵都一樣。
"""
import json, sys, tempfile
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import score as S


def _settlements(tmp_path, versions=(("stat_logit", "1.2.0"), ("stat_logit", "1.3.0"),
                                     ("always_up", "1.3.0"), ("claude", "1.3.0"))):
    rows = []
    for i in range(40):
        for m, ver in versions:
            rows.append({
                "pid": f"{m}{ver}{i}", "settled_at_utc": "2026-10-01",
                "model": m, "model_family": "x", "model_version": ver,
                "horizon": 20, "code": f"{1000+i}", "as_of": "20260918",
                "target_date": "20261015", "price_start": 10.0, "price_end": 11.0,
                "actual_return": 0.1 if i % 3 else -0.05,
                "actual_direction": 1 if i % 3 else -1,
                "predicted_direction": 1, "prob_up": 0.55,
                "correct": 1 if i % 3 else 0, "brier": 0.2, "exp_ret": 0.02,
                "in_interval": 1, "interval_score": 0.3, "market": "TW", "anchor": 0.53,
            })
    f = tmp_path / "settlements.jsonl"
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return f


def test_every_model_row_has_the_same_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SETTLEMENTS", _settlements(tmp_path))
    models = S.summary()["by_horizon"]["20"]["models"]
    assert len(models) >= 5, "同名多版本沒有被分列"
    keys = [set(m) for m in models]
    diff = [sorted(k ^ keys[0]) for k in keys if k != keys[0]]
    assert not diff, f"成績單的列 schema 不一致：{diff}"
    for need in ("edge_vs_always_up", "brier_skill", "coverage_80",
                 "interval_score", "version_split"):
        assert need in keys[0], f"缺少 {need}"


def test_versions_are_split_and_labelled(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SETTLEMENTS", _settlements(tmp_path))
    models = S.summary()["by_horizon"]["20"]["models"]
    names = {m["model"] for m in models}
    assert {"stat_logit", "stat_logit@1.2.0", "stat_logit@1.3.0"} <= names
    for m in models:
        assert m["version_split"] == ("@" in m["model"])
    # 只有一個版本的模型不分列 —— 分了只是把同一列印兩次
    assert "always_up@1.3.0" not in names


def test_brier_skill_is_against_climatology_not_always_up(tmp_path, monkeypatch):
    """always_up 固定回答 0.55，那不是基本率。只跟它比 Brier，
    模型可以只因為 0.55 ≠ 真實基本率而「贏」。

    2026-09-24 起分成兩個數：brier_skill 對**長期基本率**（config.LONGRUN_BASE_RATE，
    事前常數）；brier_skill_insample 對當批事後上漲比率（舊定義）。
    後者對任何常數預測必為負（Brier(p) = clim(1−clim) + (p−clim)²），
    只供對照，不能拿來排序。"""
    monkeypatch.setattr(S, "SETTLEMENTS", _settlements(tmp_path))
    out = S.summary()["by_horizon"]["20"]
    clim = out["base_rate_up"]
    assert abs(out["brier_climatology"] - clim * (1 - clim)) < 1e-9
    lr = S.LONGRUN_BASE_RATE[("TW", 20)]
    for m in out["models"]:
        want_in = 1 - m["brier"] / out["brier_climatology"]
        assert abs(m["brier_skill_insample"] - want_in) < 1e-3
        want_lr = 1 - m["brier"] / (lr * (1 - lr))
        assert abs(m["brier_skill"] - want_lr) < 1e-3


def test_per_market_block_has_same_schema(tmp_path, monkeypatch):
    """分市場的列與混池的列必須同 schema，混池列保留給既有下游。"""
    monkeypatch.setattr(S, "SETTLEMENTS", _settlements(tmp_path))
    out = S.summary()
    assert "TW" in out["by_horizon_market"]["20"]
    pooled = set(out["by_horizon"]["20"]["models"][0])
    per = set(out["by_horizon_market"]["20"]["TW"]["models"][0])
    assert pooled == per, sorted(pooled ^ per)


def test_report_survives_a_row_shape_it_did_not_expect(tmp_path, monkeypatch):
    """下游不該因為上游多一種列就整份報表掛掉。"""
    import report as R
    monkeypatch.setattr(S, "SETTLEMENTS", _settlements(tmp_path))
    models = S.summary()["by_horizon"]["20"]["models"]
    rows = [{"模型": m["模型"] if "模型" in m else m["model"],
             "對猜漲超額": "—" if m.get("edge_vs_always_up") is None else "x"}
            for m in models]
    assert isinstance(R._tbl(pd.DataFrame(rows), "num"), str)
