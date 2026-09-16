"""歷史回測。

刻意的隔離：回測結果寫到 data/backtest/，絕不混入 predictions.jsonl。
predictions.jsonl 只放「真實的、事前做出的」預測；一旦讓回測結果混進去，
整份記錄的可信度就毀了，而且毀得無聲無息。

回測結果該怎麼讀（三個必須同時講清楚的但書）：
  1. LLM 無法回測 —— 它知道歷史結果，任何回測分數都是假的。
  2. 統計模型雖是 walk-forward，但「哪些特徵、哪個模型」是我事後挑的，
     這個研究者自由度無法用回測消除。真實成績只能從上線日往後算。
  3. 樣本重疊：相鄰日期的 20 日預測高度重疊，有效樣本遠小於名目筆數。
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, HORIZONS
from features.build import build_all, labels_all, FEATURE_COLS
from models import baselines, statistical

OUT = DATA / "backtest"


def run(stride: int = 5, months: int = 12) -> pd.DataFrame:
    pnl = pd.read_parquet(DATA / "features/panel.parquet")
    feats_all = build_all(pnl)
    dates = sorted(pnl["date"].unique())
    start_i = max(len(dates) - months * 21, 120)
    sample = dates[start_i::stride]

    rows, t0 = [], time.time()
    for h in HORIZONS:
        labs = labels_all(pnl, h).set_index(["as_of", "code"])
        for n, d in enumerate(sample, 1):
            f = feats_all[feats_all["as_of"] == d].reset_index(drop=True)
            if f.empty:
                continue
            got = {}
            for name, fn in {**baselines.ALL, **statistical.ALL}.items():
                try:
                    o = fn(f, h, d, pnl)
                    if o is not None and not o.empty:
                        got[name] = o
                except Exception as e:  # noqa: BLE001
                    print(f"  {name} h={h} {str(d)[:10]}: {e}")
            for name, o in got.items():
                for _, r in o.iterrows():
                    key = (d, str(r["code"]))
                    if key not in labs.index:
                        continue
                    lab = labs.loc[key]
                    y = int(lab["y"])
                    prob = float(r["prob_up"])
                    rows.append({
                        "as_of": pd.Timestamp(d).strftime("%Y%m%d"), "horizon": h,
                        "model": name, "code": str(r["code"]),
                        "prob_up": prob, "direction": int(r["direction"]),
                        "actual_direction": 1 if y else -1,
                        "actual_return": float(lab["fwd_ret"]),
                        "correct": int((int(r["direction"]) == 1) == bool(y)),
                        "brier": (prob - y) ** 2,
                    })
            if n % 15 == 0:
                print(f"  h={h} {n}/{len(sample)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "results.parquet", index=False)
    return df


def report(df: pd.DataFrame) -> str:
    lines = []
    for h, gh in df.groupby("horizon"):
        base = gh[gh["model"] == "always_up"]
        base_acc = base["correct"].mean()
        n_dates = gh["as_of"].nunique()
        lines.append(f"\n{'='*78}")
        lines.append(f"  期間 {h} 個交易日   取樣日 {n_dates} 天   "
                     f"實際上漲基本率 {(gh['actual_direction']>0).mean():.1%}")
        lines.append(f"{'='*78}")
        lines.append(f"  {'模型':<12}{'筆數':>7}{'準確率':>9}{'Brier':>9}"
                     f"{'vs猜漲':>9}{'平均勝率差':>11}  {'判讀'}")
        lines.append("  " + "-" * 74)
        recs = []
        for m, g in gh.groupby("model"):
            acc = g["correct"].mean()
            edge = acc - base_acc
            # 配對比較：同 (日期,標的) 下與 always_up 的差
            j = g.merge(base[["as_of", "code", "correct"]], on=["as_of", "code"],
                        suffixes=("", "_b"))
            paired = (j["correct"] - j["correct_b"]).mean() if len(j) else np.nan
            recs.append((m, len(g), acc, g["brier"].mean(), edge, paired))
        for m, n, acc, br, edge, paired in sorted(recs, key=lambda x: x[3]):
            if m == "always_up":
                note = "← 基準線"
            elif paired > 0.02:
                note = "可能有訊號（需前瞻驗證）"
            elif paired > 0:
                note = "些微領先，統計上無意義"
            else:
                note = "無價值"
            lines.append(f"  {m:<12}{n:>7,}{acc:>8.1%}{br:>9.4f}"
                         f"{edge:>+9.1%}{paired:>+11.1%}  {note}")
    return "\n".join(lines)


if __name__ == "__main__":
    df = run()
    print(report(df))
    print(f"\n回測明細已存：{OUT/'results.parquet'}（{len(df):,} 列）")
