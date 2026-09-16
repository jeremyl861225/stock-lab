"""把 Claude 判斷檔寫入 predictions.jsonl（append-only），並存推理鏈。"""
from __future__ import annotations
import datetime as dt, hashlib, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREDICTIONS
from models import claude_judgment as cj
from predict import _existing, _append, _pid


def run(path: str) -> dict:
    p = Path(path)
    d = cj.load(p)
    fr = cj.to_frame(d)
    if fr.empty:
        return {"written": 0, "reason": "全部棄權"}

    as_of, h = d["as_of"], d["horizon"]
    seen = _existing()
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    created = dt.datetime.now(dt.UTC).isoformat()
    by_code = {j["code"]: j for j in d["judgments"]}

    recs, pid_map = [], {}
    for _, r in fr.iterrows():
        code = str(r["code"])
        pid = _pid(as_of, h, "claude", code, cj.VERSION)
        pid_map[code] = pid
        if pid in seen:
            continue
        j = by_code.get(code, {})
        recs.append({
            "pid": pid, "run_id": run_id, "created_at_utc": created,
            "as_of": as_of, "code": code, "horizon": h,
            "model": "claude", "model_family": "judgment",
            "model_version": cj.VERSION,
            "prob_up": round(float(r["prob_up"]), 6),
            "exp_ret": round(float(r["exp_ret"]), 6),
            "ret_q10": round(float(r["ret_q10"]), 6),
            "ret_q90": round(float(r["ret_q90"]), 6),
            "up_magnitude": j.get("up_magnitude"),
            "dn_magnitude": j.get("dn_magnitude"),
            "reward_risk": j.get("reward_risk"),
            "asymmetry": j.get("asymmetry"),
            "conviction": j.get("conviction"),
            "direction": int(r["direction"]),
            "rationale": str(r["rationale"])[:200],
            "feature_hash": "",
        })
    n = _append(recs)
    m = cj.save_reasoning(d, pid_map)
    return {"as_of": as_of, "horizon": h, "written": n, "reasoning_saved": m}


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(json.dumps(run(f), ensure_ascii=False))
