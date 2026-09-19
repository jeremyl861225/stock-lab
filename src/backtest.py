"""歷史回測。

刻意的隔離：回測結果寫到 data/backtest/，絕不混入 predictions.jsonl。
predictions.jsonl 只放「真實的、事前做出的」預測；一旦讓回測結果混進去，
整份記錄的可信度就毀了，而且毀得無聲無息。

回測結果該怎麼讀（三個必須同時講清楚的但書）：
  1. LLM 無法回測 —— 它知道歷史結果，任何回測分數都是假的。
  2. 統計模型雖是 walk-forward，但「哪些特徵、哪個模型」是我事後挑的，
     這個研究者自由度無法用回測消除。真實成績只能從上線日往後算。
  3. 樣本重疊：相鄰日期的 20 日預測高度重疊，有效樣本遠小於名目筆數。

2026-09-19 修掉兩個讓回測數字不可用的缺陷：
  a. **評分集合改用 point-in-time 成分股**。原本對 panel 裡的全部 67 檔台股
     逐日評分，包含「當時不在前 50、後來才漲進來」的那些 —— 那正是
     README 最上方警告的生存者偏差，而 `config/universe/` 早就有 26 份
     每月快照可用（`universe.load(as_of)`），只是回測沒接上。
  b. **兩個市場不再擠進同一個橫斷面**。原本 `feats_all[as_of == d]` 會把
     當天的台股與美股放在一起排名，於是台股的跌幅參與決定美股的名次。
     台美的交易日與漲跌互不相干，那個橫斷面沒有意義。

美股沒有歷史成分股快照（`config/universe_us/` 最早只到 2026-09-17，
晚於整個回測期間），所以美股的列一律標 `pit=False`，
`report()` 會把它們與台股分開印並標明不可引用。
造一份假的歷史快照比沒有快照更糟 —— 那是把猜測寫成資料。
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
from collect import universe

OUT = DATA / "backtest"


def _pit_members(market: str, d) -> set[str] | None:
    """該日的成分股。回傳 None 代表「沒有快照，無法還原」——
    這時不要用今天的成分股頂替，那正是生存者偏差本身。

    兩個市場的快照品質不同，讀數字時要記得（`METHOD.md` §6.2）：
      台股　0050 的每日申購買回籃子，零生存者偏差是結構上成立的。
      美股　SEC 申報股數 × 當時未還原收盤價重算市值排序。候選池是
            今天的大型股清單，已掉出池外或下市的公司還原不了 —— 弱一級。
    """
    key = pd.Timestamp(d).strftime("%Y%m%d")
    try:
        if market == "TW":
            uni = universe.load(key)
        elif market == "US":
            from collect import us as us_mod
            uni = us_mod.load(key)
        else:
            return None
    except Exception:  # noqa: BLE001
        return None
    return {c["code"] for c in uni["constituents"]}


def run(stride: int = 5, months: int = 12) -> pd.DataFrame:
    pnl_all = pd.read_parquet(DATA / "features/panel.parquet")
    markets = (sorted(pnl_all["market"].unique())
               if "market" in pnl_all.columns else ["TW"])
    rows, t0 = [], time.time()

    for market in markets:
        pnl = (pnl_all[pnl_all["market"] == market] if "market" in pnl_all.columns
               else pnl_all).copy()
        if pnl.empty:
            continue
        # 特徵與標籤都只在該市場內計算 —— 交易日曆不同，混算會把休市日
        # 的另一個市場當成「大盤」餵進來。
        feats_all = build_all(pnl)
        dates = sorted(pnl["date"].unique())
        start_i = max(len(dates) - months * 21, 120)
        sample = dates[start_i::stride]
        if not sample:
            continue
        n_pit = 0
        for h in HORIZONS:
            labs = labels_all(pnl, h).set_index(["as_of", "code"])
            for n, d in enumerate(sample, 1):
                f = feats_all[feats_all["as_of"] == d].reset_index(drop=True)
                if f.empty:
                    continue
                members = _pit_members(market, d)
                if members is not None:
                    f = f[f["code"].isin(members)].reset_index(drop=True)
                    n_pit += 1
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
                            "market": market, "pit": members is not None,
                            "model": name, "code": str(r["code"]),
                            "prob_up": prob, "direction": int(r["direction"]),
                            "actual_direction": 1 if y else -1,
                            "actual_return": float(lab["fwd_ret"]),
                            "correct": int((int(r["direction"]) == 1) == bool(y)),
                            "brier": (prob - y) ** 2,
                        })
                if n % 15 == 0:
                    print(f"  [{market}] h={h} {n}/{len(sample)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  [{market}] 取樣日 {len(sample)}　"
              f"{'PIT 成分股已套用' if n_pit else '⚠ 無歷史快照，未做 PIT 限制'}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / "results.parquet", index=False)
    return df


def report(df: pd.DataFrame) -> str:
    lines = []
    if "market" not in df.columns:
        df = df.assign(market="TW", pit=False)
    for (market, pit), gm in df.groupby(["market", "pit"]):
        tag = ("point-in-time 成分股" if pit
               else "⚠ 未做 PIT 限制，帶生存者偏差，不可引用為證據")
        for h, gh in gm.groupby("horizon"):
            base = gh[gh["model"] == "always_up"]
            base_acc = base["correct"].mean()
            n_dates = gh["as_of"].nunique()
            lines.append(f"\n{'='*78}")
            lines.append(f"  {market}　期間 {h} 個交易日　取樣日 {n_dates} 天　"
                         f"實際上漲基本率 {(gh['actual_direction']>0).mean():.1%}")
            lines.append(f"  {tag}")
            lines.append(f"{'='*78}")
            lines.append(f"  {'模型':<12}{'筆數':>7}{'準確率':>9}{'Brier':>9}"
                         f"{'vs猜漲':>9}{'平均勝率差':>11}  {'判讀'}")
            lines.append("  " + "-" * 74)
            recs = []
            for m, g in gh.groupby("model"):
                acc = g["correct"].mean()
                edge = acc - base_acc
                j = g.merge(base[["as_of", "code", "correct"]], on=["as_of", "code"],
                            suffixes=("", "_b"))
                paired = (j["correct"] - j["correct_b"]).mean() if len(j) else np.nan
                recs.append((m, len(g), acc, g["brier"].mean(), edge, paired))
            for m, n, acc, br, edge, paired in sorted(recs, key=lambda x: x[3]):
                if m == "always_up":
                    note = "← 基準線"
                elif not pit:
                    note = "樣本有偏，不判讀"
                elif pd.isna(paired):
                    note = "無配對樣本"
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
