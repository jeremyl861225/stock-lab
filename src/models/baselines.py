"""笨基準線 —— 判斷其他模型有無價值的唯一標尺。

每條線都必須同時給方向機率與期望報酬，否則無法與真實模型並排比較。
`always_up` 的期望報酬用「無條件歷史平均」，這是最誠實的笨答案：
不看任何特徵，就賭市場長期往上。要打敗它，模型得證明自己真的看懂了什麼。
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd

VERSION = "1.2.0"


def _hist_mean_ret(panel: pd.DataFrame, as_of, horizon: int) -> float:
    """as_of 之前、已實現的 horizon 期報酬平均。不得使用未來資料。"""
    hist = panel[panel["date"] <= pd.Timestamp(as_of)]
    dates = sorted(hist["date"].unique())
    if len(dates) < horizon + 2:
        return 0.0
    cutoff = dates[-(horizon + 1)]
    h = hist.sort_values(["code", "date"]).copy()
    h["fwd"] = h.groupby("code", sort=False)["close"].shift(-horizon) / h["close"] - 1
    v = h[(h["date"] <= cutoff)]["fwd"].dropna()
    return float(v.mean()) if len(v) else 0.0


def always_up(feats, horizon, as_of, panel) -> pd.DataFrame:
    """永遠猜漲，幅度用無條件歷史平均。台股長期偏多，這條線不好打敗。"""
    m = _hist_mean_ret(panel, as_of, horizon)
    n = len(feats)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": 0.55,
        "exp_ret": np.full(n, m), "ret_q10": np.full(n, m - 0.08),
        "ret_q90": np.full(n, m + 0.08), "direction": 1,
        "rationale": [f"baseline: 無條件看多（歷史均報酬 {m:+.2%}）"] * n,
    })


def random_walk(feats, horizon, as_of, panel) -> pd.DataFrame:
    """隨機猜。seed 由 (as_of, code, horizon) 決定，結果可完全重現。"""
    probs, rets = [], []
    for c in feats["code"]:
        h = hashlib.sha256(f"{as_of}|{c}|{horizon}".encode()).digest()
        p = int.from_bytes(h[:4], "big") / 2**32
        probs.append(p)
        rets.append((p - 0.5) * 0.10)
    p = np.array(probs); r = np.array(rets)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": p, "exp_ret": r,
        "ret_q10": r - 0.08, "ret_q90": r + 0.08,
        "direction": np.where(r >= 0, 1, -1),
        "rationale": ["baseline: 隨機（seed 可重現）"] * len(p),
    })


def momentum(feats, horizon, as_of, panel) -> pd.DataFrame:
    """動能延續：過去 20 日漲就猜續漲。最便宜的真實訊號。"""
    r20 = feats["ret_20"].fillna(0.0).to_numpy()
    p = np.clip(0.5 + np.tanh(r20 * 5) * 0.12, 0.05, 0.95)
    m = _hist_mean_ret(panel, as_of, horizon)
    exp = m + np.tanh(r20 * 4) * 0.03
    vol = feats["vol_20"].fillna(0.02).to_numpy() * np.sqrt(horizon)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": p, "exp_ret": exp,
        "ret_q10": exp - 1.2816 * vol, "ret_q90": exp + 1.2816 * vol,
        "direction": np.where(exp >= 0, 1, -1),
        "rationale": [f"baseline: 20日動能 {x:+.1%}" for x in r20],
    })


ALL = {"always_up": always_up, "random": random_walk, "momentum": momentum}
