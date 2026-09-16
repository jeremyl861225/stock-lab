"""每日預測主程式 —— 系統的心臟。

憲法（違反任何一條，整個系統就失去意義）：
  A. predictions.jsonl 只准 append，永不修改、永不刪除。
  B. 同一組 (as_of, horizon, model, code) 只能有一筆預測，重跑不得覆寫。
  C. 每筆預測都帶 feature_hash，日後可驗證當時真的用了那些輸入。
  D. baseline 每天必須出手，沒有例外。
"""
from __future__ import annotations
import datetime as dt, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREDICTIONS, HORIZONS, FEATURES
from collect import universe
from features import panel as panel_mod
from features.build import build as build_feats, feature_hash, FEATURE_COLS
from models import baselines, statistical, llm


def _pid(as_of: str, horizon: int, model: str, code: str, ver: str) -> str:
    return hashlib.sha256(f"{as_of}|{horizon}|{model}|{code}|{ver}".encode()).hexdigest()[:16]


def _existing() -> set[str]:
    if not PREDICTIONS.exists():
        return set()
    out = set()
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try: out.add(json.loads(line)["pid"])
            except Exception: pass  # noqa: BLE001,S110
    return out


def _append(records: list[dict]) -> int:
    if not records:
        return 0
    PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS.open("a", encoding="utf-8") as f:   # 只 append
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def run(as_of: str | None = None, horizons: list[int] | None = None,
        rebuild_panel: bool = True) -> dict:
    horizons = horizons or HORIZONS
    uni = universe.load()
    codes = {c["code"] for c in uni["constituents"]}

    if rebuild_panel or not (FEATURES / "panel.parquet").exists():
        pnl = panel_mod.build(codes)
    else:
        pnl = pd.read_parquet(FEATURES / "panel.parquet")

    as_of_ts = pd.Timestamp(as_of) if as_of else pnl["date"].max()
    as_of_str = as_of_ts.strftime("%Y%m%d")
    feats = build_feats(pnl, as_of_ts)
    if feats.empty:
        raise RuntimeError(f"{as_of_str} 無特徵可用")
    feats = feats[feats["code"].isin(codes)].reset_index(drop=True)

    seen = _existing()
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    created = dt.datetime.now(dt.UTC).isoformat()
    hashes = {r["code"]: feature_hash(r) for _, r in feats.iterrows()}
    summary = {}

    for h in horizons:
        model_fns = {
            **{k: (v, baselines.VERSION, "baseline") for k, v in baselines.ALL.items()},
            **{k: (v, statistical.VERSION, "stat") for k, v in statistical.ALL.items()},
            **{k: (v, llm.VERSION, "llm") for k, v in llm.ALL.items()},
        }
        for name, (fn, ver, family) in model_fns.items():
            try:
                if family == "llm":
                    out = fn(feats, h, as_of_ts, pnl, uni)
                else:
                    out = fn(feats, h, as_of_ts, pnl)
            except Exception as e:  # noqa: BLE001
                print(f"  [{name} h={h}] 失敗：{e}")
                continue
            if out is None or out.empty:
                print(f"  [{name} h={h}] 未出手（資料不足或未設定）")
                summary[f"{name}_h{h}"] = 0
                continue

            recs = []
            for _, r in out.iterrows():
                code = str(r["code"])
                pid = _pid(as_of_str, h, name, code, ver)
                if pid in seen:
                    continue
                seen.add(pid)
                recs.append({
                    "pid": pid, "run_id": run_id, "created_at_utc": created,
                    "as_of": as_of_str, "code": code, "horizon": h,
                    "model": name, "model_family": family, "model_version": ver,
                    "prob_up": round(float(r["prob_up"]), 6),
                    "exp_ret": round(float(r.get("exp_ret", 0.0)), 6),
                    "ret_q10": round(float(r.get("ret_q10", np.nan)), 6)
                               if pd.notna(r.get("ret_q10")) else None,
                    "ret_q90": round(float(r.get("ret_q90", np.nan)), 6)
                               if pd.notna(r.get("ret_q90")) else None,
                    "direction": int(r["direction"]),
                    "rationale": str(r.get("rationale", ""))[:200],
                    "feature_hash": hashes.get(code, ""),
                })
            n = _append(recs)
            summary[f"{name}_h{h}"] = n
            print(f"  [{name} h={h}] 新增 {n} 筆")

    return {"as_of": as_of_str, "run_id": run_id, "universe": len(codes),
            "written": summary}


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else None
    res = run(a)
    print(json.dumps(res, ensure_ascii=False, indent=2))
