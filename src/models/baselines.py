"""笨基準線 —— 判斷其他模型有無價值的唯一標尺。

若你的 LLM 或 ML 模型贏不過 always_up，它的價值就是零。
這三條線每天照常出手，永遠不准關掉。
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd

VERSION = "1.0.0"

def always_up(feats: pd.DataFrame, horizon: int, as_of) -> pd.DataFrame:
    """永遠猜漲。台股長期上漲，這條線比多數人想像的難打敗。"""
    return pd.DataFrame({
        "code": feats["code"], "prob_up": 0.55, "direction": 1,
        "rationale": "baseline: 無條件看多",
    })

def random_walk(feats: pd.DataFrame, horizon: int, as_of) -> pd.DataFrame:
    """隨機猜。用 (as_of, code, horizon) 做 seed，結果可完全重現。"""
    probs = []
    for c in feats["code"]:
        h = hashlib.sha256(f"{as_of}|{c}|{horizon}".encode()).digest()
        probs.append(int.from_bytes(h[:4], "big") / 2**32)
    p = np.array(probs)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": p,
        "direction": np.where(p >= 0.5, 1, -1),
        "rationale": "baseline: 隨機（seed 可重現）",
    })

def momentum(feats: pd.DataFrame, horizon: int, as_of) -> pd.DataFrame:
    """動能延續：過去 20 日漲就猜續漲。最便宜的真實訊號。"""
    r = feats["ret_20"].fillna(0.0)
    p = (0.5 + np.tanh(r * 5) * 0.12).clip(0.05, 0.95)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": p.values,
        "direction": np.where(p >= 0.5, 1, -1),
        "rationale": ["baseline: 20日動能 " + f"{x:+.1%}" for x in r],
    })

ALL = {"always_up": always_up, "random": random_walk, "momentum": momentum}
