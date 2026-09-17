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


def _pid(as_of: str, horizon: int, model: str, code: str, ver: str = "") -> str:
    """身分鍵刻意不含 model_version。

    原本含版本，等於替同一組 (as_of, horizon, model, code) 重開一個槽 ——
    違反本檔憲法 B，而且已經發生：改模型→重跑→新版本→兩筆都被結算，
    等於同一天對同一檔下兩次注，只要有一次對就進帳。這正是系統要防的自我欺騙。
    版本改以欄位記錄，計分時一律取每組最新的一筆。"""
    return hashlib.sha256(f"{as_of}|{horizon}|{model}|{code}".encode()).hexdigest()[:16]


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
    # 美股 universe 一併納入，否則自動化只會預測台股
    try:
        from collect import us as us_mod
        uni_us = us_mod.load()
        codes |= {c["code"] for c in uni_us["constituents"]}
        uni = {**uni, "constituents": uni["constituents"] + uni_us["constituents"]}
    except Exception as e:  # noqa: BLE001
        print(f"  美股 universe 未載入（{e}），僅預測台股")

    if rebuild_panel or not (FEATURES / "panel.parquet").exists():
        # panel 要涵蓋「曾入選過」的全集；預測範圍另由下方 isin(codes) 限定在今日成分股。
        # 只用今日成分股建 panel，會把後來掉出前 50 的股票從 panel 整批抹掉 ——
        # 它們尚未到期的預測在 settle 就永遠找不到價格（missing），靜默消失。
        # 掉出成分股的多半是跌下去的那些，正是模型看錯的樣本，成績單會因此偏樂觀。
        pnl = panel_mod.build(set(universe.codes_ever()) | codes)
    else:
        pnl = pd.read_parquet(FEATURES / "panel.parquet")

    as_of_ts = pd.Timestamp(as_of) if as_of else pnl["date"].max()
    as_of_str = as_of_ts.strftime("%Y%m%d")
    feats = build_feats(pnl, as_of_ts)
    # 每個市場的最新交易日不同（台股 13:30 收盤、美股 21:30 才開盤），
    # build() 已逐市場算好各自的 as_of。若在此用全域 max 覆蓋，
    # 美股會被標上台股的日期 → settle 永遠找不到對應交易日 → 永久失聯。
    # 實測：settled 50、missing 53。台股休市日則反向失聯。
    feats["as_of_str"] = pd.to_datetime(feats["as_of"]).dt.strftime("%Y%m%d")
    if feats.empty:
        raise RuntimeError(f"{as_of_str} 無特徵可用")
    feats = feats[feats["code"].isin(codes)].reset_index(drop=True)

    seen = _existing()
    run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    created = dt.datetime.now(dt.UTC).isoformat()
    hashes = {r["code"]: feature_hash(r) for _, r in feats.iterrows()}
    code_as_of = dict(zip(feats["code"], feats["as_of_str"]))
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
                a_of = code_as_of.get(code, as_of_str)   # 逐檔用自己市場的交易日
                pid = _pid(a_of, h, name, code, ver)
                if pid in seen:
                    continue
                seen.add(pid)
                recs.append({
                    "pid": pid, "run_id": run_id, "created_at_utc": created,
                    "as_of": a_of, "code": code, "horizon": h,
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
