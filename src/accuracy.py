# -*- coding: utf-8 -*-
"""期望值加權準確率。

為什麼要加權：命中率把「期望值 +4.9% 的南亞科」和「期望值 +0.1% 的統一」
算成同樣一票，但後者本來就等於沒有觀點。加權後，猜得越篤定、賭注越大的
那些判斷，對分數的影響才越大 —— 這才對得起「期望值排序」這個用法。

    加權準確率 = Σ(|期望值| × 命中) / Σ|期望值|

另外一併算「期望值 vs 實際報酬」的迴歸斜率：斜率接近 1 代表幅度估得準，
遠小於 1 代表方向可能對但幅度灌水。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SETTLEMENTS, PREDICTIONS


def _jsonl(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    rows = []
    for l in p.read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def summary(model: str = "claude") -> dict:
    s = _jsonl(SETTLEMENTS)
    preds = _jsonl(PREDICTIONS)
    out = {"status": "pending", "settled": 0, "pending": 0, "first_due": None}

    if not preds.empty:
        p = preds[preds["model"] == model]
        out["pending"] = int(len(p))
    if s.empty or model not in set(s.get("model", [])):
        return out

    d = s[s["model"] == model].copy()
    d = d.sort_values("settled_at_utc").drop_duplicates(
        ["as_of", "horizon", "code"], keep="last")
    if d.empty or "exp_ret" not in d:
        return out

    d["exp_ret"] = pd.to_numeric(d["exp_ret"], errors="coerce")
    d = d.dropna(subset=["exp_ret", "actual_return"])
    if d.empty:
        return out

    w = d["exp_ret"].abs()
    res = {"status": "ok", "settled": int(len(d)), "pending": out["pending"],
           "hit_rate": float(d["correct"].mean()),
           "weighted_hit": float((w * d["correct"]).sum() / w.sum()) if w.sum() else np.nan,
           "by_horizon": {}}
    # 幅度校準：實際 = a + b × 預測，b 接近 1 才代表幅度估得準
    if len(d) > 2 and d["exp_ret"].std() > 0:
        b, a = np.polyfit(d["exp_ret"], d["actual_return"], 1)
        res["calib_slope"] = round(float(b), 3)
    for h, g in d.groupby("horizon"):
        wg = g["exp_ret"].abs()
        res["by_horizon"][str(h)] = {
            "n": int(len(g)), "hit_rate": round(float(g["correct"].mean()), 4),
            "weighted_hit": round(float((wg * g["correct"]).sum() / wg.sum()), 4)
                            if wg.sum() else None,
            "mean_pred": round(float(g["exp_ret"].mean()), 5),
            "mean_actual": round(float(g["actual_return"].mean()), 5),
        }
    return res


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=2))
