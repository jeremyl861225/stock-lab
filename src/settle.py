"""結算：把到期的預測拿去對真實答案。

刻意的設計：結算結果寫進另一個 append-only 檔（settlements.jsonl），
不回頭修改 predictions.jsonl。兩個檔案用 pid 關聯。
這樣「預測」與「結果」在檔案層級就是分離且不可互相污染的。

target_date 不在預測當下寫死，而是結算時從實際交易日曆往後數 horizon 天 ——
預測當下並不知道未來哪幾天有開市（颱風假、補班日），寫死只會製造假資料。
"""
from __future__ import annotations
import datetime as dt, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PREDICTIONS, SETTLEMENTS, FEATURES


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try: out.append(json.loads(line))
            except Exception: pass  # noqa: BLE001,S110
    return out


def run() -> dict:
    preds = _load_jsonl(PREDICTIONS)
    if not preds:
        return {"settled": 0, "reason": "尚無預測"}
    done = {s["pid"] for s in _load_jsonl(SETTLEMENTS)}
    pnl = pd.read_parquet(FEATURES / "panel.parquet")
    pnl["ds"] = pnl["date"].dt.strftime("%Y%m%d")

    by_code = {c: g.sort_values("date").reset_index(drop=True)
               for c, g in pnl.groupby("code", sort=False)}
    latest = pnl["date"].max()

    recs, pending, missing = [], 0, 0
    for p in preds:
        if p["pid"] in done:
            continue
        sub = by_code.get(p["code"])
        if sub is None:
            missing += 1
            continue
        idx = sub.index[sub["ds"] == p["as_of"]]
        if len(idx) == 0:
            missing += 1
            continue
        i = int(idx[0])
        j = i + p["horizon"]
        if j >= len(sub):
            pending += 1          # 還沒到期，下次再結算
            continue
        p0 = float(sub.loc[i, "close"])
        p1 = float(sub.loc[j, "close"])
        if not (p0 > 0 and p1 > 0):
            missing += 1
            continue
        ret = p1 / p0 - 1
        actual = 1 if ret > 0 else -1
        prob = float(p["prob_up"])
        y = 1 if ret > 0 else 0
        recs.append({
            "pid": p["pid"], "settled_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "model": p["model"], "model_family": p.get("model_family", ""),
            "horizon": p["horizon"], "code": p["code"], "as_of": p["as_of"],
            "target_date": sub.loc[j, "ds"], "price_start": p0, "price_end": p1,
            "actual_return": round(ret, 6), "actual_direction": actual,
            "predicted_direction": int(p["direction"]), "prob_up": prob,
            "correct": int(p["direction"] == actual),
            "brier": round((prob - y) ** 2, 6),
            # 幅度面：方向對但幅度錯的模型一樣沒有用
            "exp_ret": p.get("exp_ret"),
            "ret_error": (None if p.get("exp_ret") is None
                          else round(ret - float(p["exp_ret"]), 6)),
            "in_interval": (None if p.get("ret_q10") is None or p.get("ret_q90") is None
                            else int(float(p["ret_q10"]) <= ret <= float(p["ret_q90"]))),
        })

    if recs:
        with SETTLEMENTS.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"settled": len(recs), "pending": pending, "missing": missing,
            "latest_data": latest.strftime("%Y%m%d")}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
