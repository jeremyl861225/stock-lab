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

    # 既有列按 pid 收攏。修訂要能分辨兩件事：
    #   (a) 真的改了判斷 —— 該留下痕跡，且修訂編號要遞增
    #   (b) 只是重跑重新匯入（面板重建後波動度微幅漂移）—— 不該留痕跡
    # 舊版兩者都寫成 revision=1，在帳本裡長得一模一樣，等於讓「改判斷」
    # 可以偽裝成「重跑」。實測 20260916 的美股 53 檔各被寫了 3 次。
    prior = {}
    if PREDICTIONS.exists():
        for line in PREDICTIONS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            prior.setdefault(r["pid"], []).append(r)

    def _material(old: dict, new: dict) -> bool:
        """判斷是否為實質修訂。純重跑造成的浮點漂移不算。"""
        if round(old.get("prob_up") or 0, 3) != round(new["prob_up"], 3):
            return True
        for k in ("up_magnitude", "dn_magnitude"):
            a_, b_ = old.get(k), new.get(k)
            if a_ is None or b_ is None:
                if a_ != b_:
                    return True
                continue
            # 幅度相對變動 >1% 才算改了判斷；波動度重算的漂移約 0.3%
            if abs(b_) > 1e-9 and abs(a_ / b_ - 1) > 0.01:
                return True
        return False

    recs, pid_map, skipped, noop = [], {}, [], []
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
            "revision": len(prior.get(pid, [])),
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
    # 把沒有實質改變的修訂剔掉，不留無意義的列
    if revise:
        keep = []
        for rec in recs:
            old = prior.get(rec["pid"])
            if old and not _material(old[-1], rec):
                noop.append(rec["code"])
                continue
            keep.append(rec)
        recs = keep

    n = _append(recs)
    m = cj.save_reasoning(d, pid_map)
    out = {"as_of": as_of, "horizon": h, "written": n, "reasoning_saved": m}
    if noop:
        out["noop_revisions"] = len(noop)
        print(f"  · {len(noop)} 檔的修訂與既有判斷實質相同（僅浮點漂移），未寫入")
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
