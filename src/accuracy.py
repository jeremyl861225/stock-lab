# -*- coding: utf-8 -*-
"""期望值加權準確率。

為什麼要加權：命中率把「期望值 +4.9% 的南亞科」和「期望值 +0.1% 的統一」
算成同樣一票，但後者本來就等於沒有觀點。加權後，猜得越篤定、賭注越大的
那些判斷，對分數的影響才越大 —— 這才對得起「期望值排序」這個用法。

    加權準確率 = Σ(|期望值| × 命中) / Σ|期望值|

幅度校準（2026-09-24 改）：
  舊版印一個 `calib_slope`（實際報酬對期望值的單日橫斷面迴歸斜率）。
  首批算出 −0.746，面板直接顯示「幅度校準斜率 −0.746」—— 但那是 r=−0.043
  乘上 sd 比 17.3，bootstrap 95% CI 是 [−6.6, +4.6]，留一剔除就從 −1.43 擺到 +0.93。
  單日的斜率只有 1 個獨立的市場觀測，而 METHOD §1.3 寫明要拿 Mincer–Zarnowitz
  斜率決定要不要校正 0.85 係數 —— 用它會做錯決定。
  現在拆成兩層：
    市場層  mean_pred vs mean_actual（首批 +0.07% vs +4.55%）—— 整批平移對不對
    選股層  逐日對 (exp_ret, actual) 各自去均值後 pooled 迴歸（等價於日固定效應），
            SE 以日為群集並乘 √(h/stride) 折減；有效天數 < MIN_EFF_DAYS 時斜率輸出 None，
            另附去均值後的 Spearman rank IC（對平移不變，單日也有定義）。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SETTLEMENTS, PREDICTIONS
from score import is_abstain, cluster_t, _overlap, MIN_EFF_DAYS


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


def mz_stock_layer(d: pd.DataFrame, h: int) -> dict:
    """選股層的 Mincer–Zarnowitz：逐日去均值後 y = b·x 的 pooled 斜率。

    回傳 slope／t_adj／rank_ic／n_days／n_eff_days。斜率在有效天數不足時為 None，
    rank IC 則照給（它在單日也有定義，只是要自己看 n_eff）。
    """
    d = d.dropna(subset=["exp_ret", "actual_return"]).copy()
    if d.empty:
        return {"slope": None, "t_adj": None, "rank_ic": None, "n_days": 0, "n_eff_days": 0.0}
    d["x"] = d["exp_ret"] - d.groupby("as_of")["exp_ret"].transform("mean")
    d["y"] = d["actual_return"] - d.groupby("as_of")["actual_return"].transform("mean")
    n_days = int(d["as_of"].nunique())
    overlap = _overlap(d, h)
    n_eff = round(n_days / overlap, 1)
    ics = []
    for _, g in d.groupby("as_of"):
        if g["x"].std() > 0 and g["y"].std() > 0:
            ic = g["x"].corr(g["y"], method="spearman")
            if pd.notna(ic):
                ics.append(float(ic))
    rank_ic = round(float(np.mean(ics)), 4) if ics else None
    sxx = float((d["x"] ** 2).sum())
    if sxx <= 0 or n_eff < MIN_EFF_DAYS:
        return {"slope": None, "t_adj": None, "rank_ic": rank_ic,
                "n_days": n_days, "n_eff_days": n_eff}
    b = float((d["x"] * d["y"]).sum() / sxx)
    d["u"] = d["y"] - b * d["x"]
    # 日群集 SE：Σ_day (Σ_i x_i u_i)² / (Σ x²)²，再乘 √overlap 折減
    s = d.assign(xu=d["x"] * d["u"]).groupby("as_of")["xu"].sum()
    se = math.sqrt(float((s ** 2).sum())) / sxx * math.sqrt(overlap)
    t = b / se if se > 0 else None
    return {"slope": round(b, 3), "t_adj": None if t is None else round(t, 2),
            "rank_ic": rank_ic, "n_days": n_days, "n_eff_days": n_eff}


def summary(model: str = "claude") -> dict:
    s = _jsonl(SETTLEMENTS)
    preds = _jsonl(PREDICTIONS)
    out = {"status": "pending", "settled": 0, "pending": 0, "pending_1y": 0, "first_due": None}

    settled_pids = set(s["pid"]) if not s.empty and "pid" in s.columns else set()
    if not preds.empty:
        p = preds[preds["model"] == model]
        # 必須與 settle 用同一套去重，否則會把永遠不會被結算的舊版本也算進來
        # ——實測 306 筆裡有 100 筆是舊版，虛胖 49%。
        if not p.empty:
            p = p.sort_values("created_at_utc").drop_duplicates(
                ["as_of", "horizon", "code"], keep="last")
            p = p[~p["pid"].isin(settled_pids)]          # 已結算的不算待結算
            out["pending"] = int((p["horizon"] < 250).sum())
            out["pending_1y"] = int((p["horizon"] >= 250).sum())
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

    # 中性判斷（p=0.5、期望值 0）不進命中率；權重為 0 所以加權命中率本來就不受影響
    d["abstain"] = is_abstain(d["prob_up"], d["exp_ret"]).to_numpy()
    sc = d[~d["abstain"]]
    w = sc["exp_ret"].abs()
    res = {"status": "ok", "settled": int(len(d)), "scored": int(len(sc)),
           "abstain": int(d["abstain"].sum()),
           "pending": out["pending"], "pending_1y": out["pending_1y"],
           "hit_rate": float(sc["correct"].mean()) if len(sc) else None,
           "weighted_hit": float((w * sc["correct"]).sum() / w.sum()) if w.sum() else None,
           "by_horizon": {}}
    for h, g in d.groupby("horizon"):
        gs = g[~g["abstain"]]
        wg = gs["exp_ret"].abs()
        res["by_horizon"][str(h)] = {
            "n": int(len(g)), "n_scored": int(len(gs)),
            "hit_rate": round(float(gs["correct"].mean()), 4) if len(gs) else None,
            "weighted_hit": round(float((wg * gs["correct"]).sum() / wg.sum()), 4)
                            if wg.sum() else None,
            # 市場層：整批平移對不對
            "mean_pred": round(float(g["exp_ret"].mean()), 5),
            "mean_actual": round(float(g["actual_return"].mean()), 5),
            # 選股層：去均值後的斜率與 rank IC
            "mz": mz_stock_layer(g, int(h)),
        }
    return res


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=2))


def by_code(model: str = "claude") -> dict[str, dict]:
    """每檔股票的歷史預測成績。

    個股層級的樣本本來就少（一檔一天最多兩筆），所以這個數字要很久才有意義。
    未達門檻時回傳 None，讓面板顯示「—」而不是拿 1/1 = 100% 去誤導。
    中性判斷不進命中率（但進 n，讓「—」的門檻誠實反映出手次數）。
    """
    s = _jsonl(SETTLEMENTS)
    if s.empty or "model" not in s.columns:
        return {}
    d = s[s["model"] == model].copy()
    if d.empty:
        return {}
    d = d.sort_values("settled_at_utc").drop_duplicates(
        ["as_of", "horizon", "code"], keep="last")
    d["exp_ret"] = pd.to_numeric(d.get("exp_ret"), errors="coerce")
    d["abstain"] = is_abstain(d["prob_up"], d["exp_ret"]).to_numpy()
    out = {}
    for code, g in d.groupby("code"):
        gs = g[~g["abstain"]]
        n = len(gs)
        w = gs["exp_ret"].abs()
        out[str(code)] = {
            "n": int(n),
            "hit": float(gs["correct"].mean()) if n else None,
            "weighted": (float((w * gs["correct"]).sum() / w.sum())
                         if n and w.notna().any() and w.sum() else None),
            # 樣本太少時不給數字 —— 3 筆裡對 2 筆不代表 67% 的準確率
            "reliable": bool(n >= 8),
        }
    return out
