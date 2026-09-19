"""計分：判斷模型到底有沒有價值。

核心觀念（照重要性排序，多數人只看第一項就是問題所在）：
  1. 準確率  —— 最直覺，但單獨看幾乎無意義（台股長期偏多，猜漲就有 5 成多）。
  2. 對 baseline 的超額 —— 真正該看的。贏不過 always_up 就是沒有價值。
  3. Brier score —— 同時衡量方向與信心，越低越好。
  4. 校準曲線 —— 說 70% 的那批，是否真有 70% 上漲。
  5. 統計顯著性 —— 樣本不夠就不要宣稱有 edge。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SETTLEMENTS


def _load() -> pd.DataFrame:
    if not SETTLEMENTS.exists():
        return pd.DataFrame()
    rows = []
    for l in SETTLEMENTS.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        try:                      # 一行壞掉不該讓整份成績單掛掉
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


def _binom_p(k: int, n: int, p0: float) -> float:
    """單尾二項檢定：k 次命中 / n 次，虛無假設命中率 = p0。"""
    if n == 0:
        return 1.0
    z = (k / n - p0) / math.sqrt(max(p0 * (1 - p0) / n, 1e-12))
    return 0.5 * math.erfc(z / math.sqrt(2))


def calibration(df: pd.DataFrame, bins: int = 10) -> list[dict]:
    if df.empty:
        return []
    d = df.copy()
    d["y"] = (d["actual_direction"] > 0).astype(int)
    d["bin"] = pd.cut(d["prob_up"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = []
    for b, g in d.groupby("bin", observed=True):
        out.append({"bin": str(b), "mid": float(b.mid), "n": int(len(g)),
                    "predicted": float(g["prob_up"].mean()),
                    "actual": float(g["y"].mean())})
    return out


def summary() -> dict:
    df = _load()
    if df.empty:
        return {"status": "no_settlements", "rows": 0}

    res = {"rows": int(len(df)),
           "date_range": [df["as_of"].min(), df["target_date"].max()],
           "by_horizon": {}}

    for h, gh in df.groupby("horizon"):
        base = gh[gh["model"] == "always_up"]
        base_acc = float(base["correct"].mean()) if len(base) else float("nan")
        # 氣候學基準：這一期別實際的上漲比率。Brier 技能分數 = 1 − Brier/Brier_clim，
        # 其中 Brier_clim = clim·(1−clim) 是「永遠回答基本率」的 Brier。
        # always_up 固定回答 0.55，不是基本率；只跟它比 Brier，模型可以只因為
        # 0.55 ≠ 真實基本率而「贏」。技能分數 > 0 才代表比基本率多知道一點什麼。
        clim = float((gh["actual_direction"] > 0).mean())
        brier_clim = clim * (1 - clim)
        models = []
        for m, g in gh.groupby("model"):
            n, k = len(g), int(g["correct"].sum())
            acc = k / n if n else float("nan")
            brier = float(g["brier"].mean())
            cov = (float(g["in_interval"].dropna().mean())
                   if "in_interval" in g and g["in_interval"].notna().any() else None)
            iscore = (float(g["interval_score"].dropna().mean())
                      if "interval_score" in g and g["interval_score"].notna().any() else None)
            # 與 always_up 相同 (as_of, code) 配對比較，避免樣本不同造成假差距
            paired = None
            if len(base) and m != "always_up":
                j = g.merge(base[["as_of", "code", "correct"]], on=["as_of", "code"],
                            suffixes=("", "_base"))
                if len(j):
                    paired = float(j["correct"].mean() - j["correct_base"].mean())
            models.append({
                "model": m, "family": g["model_family"].iloc[0], "n": n,
                "accuracy": round(acc, 4),
                "brier": round(brier, 4),
                "brier_skill": (round(1 - brier / brier_clim, 4) if brier_clim > 0 else None),
                # 名目 80% 區間：覆蓋率要看，但只看它會獎勵開得過寬的區間，
                # 所以並列 interval_score（含寬度罰項，越低越好）。
                "coverage_80": None if cov is None else round(cov, 4),
                "interval_score": None if iscore is None else round(iscore, 5),
                "mean_prob": round(float(g["prob_up"].mean()), 4),
                "edge_vs_always_up": None if paired is None else round(paired, 4),
                "p_value_vs_base": round(_binom_p(k, n, base_acc), 4)
                                   if not math.isnan(base_acc) else None,
                "avg_return_when_long": round(float(
                    g[g["predicted_direction"] == 1]["actual_return"].mean()), 5)
                    if (g["predicted_direction"] == 1).any() else None,
                "calibration": calibration(g),
            })
        models.sort(key=lambda x: x["brier"])
        res["by_horizon"][str(h)] = {
            "base_rate_up": round(clim, 4),
            "brier_climatology": round(brier_clim, 4),
            "models": models,
        }
    return res


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=2))
