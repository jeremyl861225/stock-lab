"""把 Claude 判斷檔寫入 predictions.jsonl（append-only），並存推理鏈。"""
from __future__ import annotations
import datetime as dt, hashlib, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREDICTIONS
from models import claude_judgment as cj
from predict import _existing, _append, _pid


def run(path: str, revise: bool = False) -> dict:
    """revise=True 時，同一身分鍵允許再寫一筆修正版。

    預設不允許（憲法 B：同一組 (as_of, horizon, model, code) 只能有一筆），
    這是為了防止反覆改判。但審核查出實質錯誤時必須能更正 ——
    此前的行為是「遇到既有 pid 直接 skip」，修正會被靜默吞掉、不報錯也不留紀錄。
    修正版以 revision 遞增並保留舊筆；settle 與 ranking 都取最新一筆計分。
    """
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

    recs, pid_map, skipped = [], {}, []
    for _, r in fr.iterrows():
        code = str(r["code"])
        pid = _pid(as_of, h, "claude", code, cj.VERSION)
        pid_map[code] = pid
        if pid in seen and not revise:
            skipped.append(code)
            continue
        j = by_code.get(code, {})
        recs.append({
            "pid": pid, "run_id": run_id, "created_at_utc": created,
            "as_of": as_of, "code": code, "horizon": h,
            "model": "claude", "model_family": "judgment",
            "model_version": cj.VERSION,
            "revision": int(revise),
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
    out = {"as_of": as_of, "horizon": h, "written": n, "reasoning_saved": m}
    if skipped:
        # 靜默跳過是最危險的行為 —— 修正會消失而沒有任何訊號
        out["skipped_existing"] = len(skipped)
        out["hint"] = "這些標的已有判斷；若這是修正版，請加 --revise"
        print(f"  ！{len(skipped)} 檔已有判斷被跳過：{skipped[:8]}"
              f"{'...' if len(skipped) > 8 else ''}")
        print(f"  若這是修正版，請執行：python src/ingest_judgment.py --revise {path}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    revise = "--revise" in args
    for f in [a for a in args if not a.startswith("--")]:
        print(json.dumps(run(f, revise), ensure_ascii=False))
