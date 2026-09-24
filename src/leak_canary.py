# -*- coding: utf-8 -*-
"""外洩金絲雀與誠實天花板 —— 回答「什麼樣的回測數字才是可信的」。

背景（2026-09-24）：使用者要求「改善到歷史回測準確率 >90%」。
5／20 日方向預測在無外洩下做不到這件事，而且不是「還沒找到方法」，是資訊集的天花板本身就低於 90%：

  誠實 stat_logit（PIT、12 個月）      TW5 51.7%  TW20 60.1%  US5 50.2%  US20 50.2%
  完美市場擇時（每日知市場正負）        62.9%      64.8%       60.7%      60.3%
  完美相對強弱（知每檔去均值正負）      80.3%      76.6%       86.4%      86.2%
  把 fwd_ret 本身當特徵                 98.7%      99.6%       100%       100%
  sign(fwd_ret)+N(0,1)（訊噪比 1）      78.6%      77.0%       82.8%      81.4%

要到 90%，只有直接知道未來報酬幅度，或注入訊噪比約 1.3 以上的標籤外洩；
誠實特徵的 rank IC 只有 0.00–0.02，連訊噪比 0.2 的外洩（IC 0.10–0.22）都不到。

所以這支程式做三件事：
  1. oracles()：各種「神諭」的準確率 —— 天花板。
  2. canaries()：把已知的外洩故意注入 walk-forward 模型，量每一種外洩「買到」多少準確率。
  3. guard()：把回測結果對照天花板 —— 任何模型的方向準確率超過「完美相對強弱（中位數版）」
     或 |rank IC| > IC_CEILING，就標成「外洩嫌疑」。

⚠ guard 是**煙霧偵測器，不是定理**。天花板取決於怎麼去均值（減均值 77–86%、減中位數 85–88%），
而同時握有部分幅度資訊的誠實模型理論上可以越過「只知正負號」的那條線。
它抓得到的是 fwd_ret 直接混進特徵那種（99%）；訊噪比 0.5 的外洩只有 63–69%，它抓不到，
那一層要靠 tests/test_no_lookahead.py 與 tests/test_panel_availability.py。
用法：
    ./.venv/bin/python src/leak_canary.py            # 跑神諭＋金絲雀，寫 data/backtest/leak_canary.json
    ./.venv/bin/python src/leak_canary.py --guard    # 只對照既有回測結果（快）
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, HORIZONS
from features.build import build_all, labels_all, FEATURE_COLS
from models import statistical
from backtest import _pit_members

OUT = DATA / "backtest" / "leak_canary.json"
IC_CEILING = 0.30            # 誠實特徵的 rank IC 只有 0.00–0.02；超過 0.3 幾乎只可能是外洩
CANARY_NOISE_SD = 1.0        # 訊噪比 1 的標籤外洩，對照用


def _sample_dates(dates, months: int, stride: int):
    start_i = max(len(dates) - months * 21, 120)
    return dates[start_i::stride]


def _pit_frame(labs: pd.DataFrame, market: str, d) -> pd.DataFrame:
    g = labs[labs["as_of"] == d]
    mem = _pit_members(market, d)
    if mem is not None:
        g = g[g["code"].isin(mem)]
    return g


def oracles(panel: pd.DataFrame, months: int = 12, stride: int = 5) -> list[dict]:
    """神諭天花板。每個 (market, h) 一列。"""
    rows = []
    for market in sorted(panel["market"].unique()):
        pm = panel[panel["market"] == market]
        dates = sorted(pm["date"].unique())
        for h in HORIZONS:
            labs = labels_all(pm, h)
            parts = [_pit_frame(labs, market, d) for d in _sample_dates(dates, months, stride)]
            g = pd.concat([p for p in parts if not p.empty])
            g = g.copy()
            g["mkt_mean"] = g.groupby("as_of")["fwd_ret"].transform("mean")
            g["mkt_med"] = g.groupby("as_of")["fwd_ret"].transform("median")
            y = g["y"] == 1
            rows.append({
                "market": market, "horizon": h, "days": int(g["as_of"].nunique()), "n": int(len(g)),
                "always_up": round(float(y.mean()), 4),
                "perfect_market_timing": round(float(((g["mkt_mean"] > 0) == y).mean()), 4),
                "perfect_relative_strength_mean": round(float(((g["fwd_ret"] - g["mkt_mean"] > 0) == y).mean()), 4),
                "perfect_relative_strength_median": round(float(((g["fwd_ret"] - g["mkt_med"] > 0) == y).mean()), 4),
                "perfect_with_magnitude": 1.0,
            })
    return rows


def _walk_forward(panel: pd.DataFrame, market: str, h: int, months: int, stride: int,
                  inject) -> tuple[float, float]:
    """照 statistical._training_set 的截斷規則做 walk-forward logit，回傳 (準確率, rank IC)。
    inject(feats, labs) -> feats：把外洩塞進特徵。訓練列也一併注入，否則模型學不到它。"""
    pm = panel[panel["market"] == market]
    dates = sorted(pm["date"].unique())
    feats_all = build_all(pm)
    labs = labels_all(pm, h)
    hits, ics = [], []
    for d in _sample_dates(dates, months, stride):
        hist = pm[pm["date"] <= d]
        hd = sorted(hist["date"].unique())
        if len(hd) < h + 80:
            continue
        cutoff = hd[-(h + 1)]
        tr = feats_all[feats_all["as_of"] <= cutoff].merge(
            labs[["as_of", "code", "y", "fwd_ret"]], on=["as_of", "code"], how="inner")
        keep = sorted(tr["as_of"].unique())[-400:][::2]
        tr = tr[tr["as_of"].isin(keep)].dropna(subset=["y"])
        te = _pit_frame(labs, market, d).merge(feats_all[feats_all["as_of"] == d],
                                               on=["as_of", "code"], how="inner")
        if len(tr) < 2000 or te.empty or tr["y"].nunique() < 2:
            continue
        tr, te = inject(tr), inject(te)
        cols = [c for c in FEATURE_COLS + ["_leak"] if c in tr.columns
                and tr[c].notna().mean() >= statistical.MIN_FEATURE_COVERAGE]
        clf = statistical._clf("logit", tr[cols], tr["y"])
        prob = clf.predict_proba(te[cols])[:, 1]
        hits.append(float(((prob > 0.5) == (te["y"] == 1)).mean()))
        if len(te) >= 10:
            ic = pd.Series(prob).corr(te["fwd_ret"].reset_index(drop=True), method="spearman")
            if pd.notna(ic):
                ics.append(float(ic))
    return (float(np.mean(hits)) if hits else float("nan"),
            float(np.mean(ics)) if ics else float("nan"))


def canaries(panel: pd.DataFrame, months: int = 12, stride: int = 5, seed: int = 0) -> list[dict]:
    """把已知外洩注入 walk-forward 模型。每個 (market, h) 一列，欄位是各種外洩買到的準確率。"""
    rng = np.random.default_rng(seed)
    kinds = {
        "honest": lambda f: f,
        "leak_fwd_ret": lambda f: f.assign(_leak=f["fwd_ret"]),
        "leak_sign_noise_sd1": lambda f: f.assign(
            _leak=np.sign(f["fwd_ret"]) + rng.normal(0, CANARY_NOISE_SD, len(f))),
        "leak_market_direction": lambda f: f.assign(
            _leak=np.sign(f.groupby("as_of")["fwd_ret"].transform("mean"))),
    }
    rows, t0 = [], time.time()
    for market in sorted(panel["market"].unique()):
        for h in HORIZONS:
            rec = {"market": market, "horizon": h}
            for k, fn in kinds.items():
                acc, ic = _walk_forward(panel, market, h, months, stride, fn)
                rec[k] = round(acc, 4)
                rec[f"{k}_rank_ic"] = round(ic, 4)
                print(f"  {market} h={h} {k:<24} acc {acc:.3f}  IC {ic:+.3f}  ({time.time()-t0:.0f}s)",
                      flush=True)
            rows.append(rec)
    return rows


def guard(results: pd.DataFrame, oracle_rows: list[dict], ic_ceiling: float = IC_CEILING) -> list[dict]:
    """把回測結果對照天花板。回傳可疑的 (market, h, model) 清單；空清單＝沒有煙。"""
    ceil = {(r["market"], int(r["horizon"])): r for r in oracle_rows}
    flags = []
    if results.empty:
        return flags
    if "pit" in results.columns:
        results = results[results["pit"]]
    for (mk, h, m), g in results.groupby(["market", "horizon", "model"]):
        c = ceil.get((mk, int(h)))
        if not c:
            continue
        acc = float(g["correct"].mean())
        ics = []
        for _, gd in g.groupby("as_of"):
            if len(gd) >= 10 and gd["prob_up"].nunique() >= 3:
                ic = gd["prob_up"].corr(gd["actual_return"], method="spearman")
                if pd.notna(ic):
                    ics.append(float(ic))
        ic = float(np.mean(ics)) if ics else 0.0
        limit = c["perfect_relative_strength_median"]
        if acc > limit or abs(ic) > ic_ceiling:
            flags.append({"market": mk, "horizon": int(h), "model": m, "accuracy": round(acc, 4),
                          "rank_ic": round(ic, 4), "ceiling_accuracy": limit,
                          "reason": ("準確率超過完美相對強弱天花板" if acc > limit
                                     else f"|rank IC| > {ic_ceiling}")})
    return flags


def report(orc: list[dict], can: list[dict], flags: list[dict]) -> str:
    L = ["", "=" * 96, "  什麼樣的回測數字才是可信的（PIT 成分股，方向準確率）", "=" * 96]
    L.append(f"  {'市場':<4}{'h':>3}{'猜漲':>8}{'誠實logit':>10}{'市場神諭':>9}{'相對強弱(均值)':>14}"
             f"{'相對強弱(中位)':>14}{'fwd_ret當特徵':>13}{'訊噪比1':>9}")
    cm = {(r["market"], r["horizon"]): r for r in can}
    for r in orc:
        c = cm.get((r["market"], r["horizon"]), {})
        L.append(f"  {r['market']:<4}{r['horizon']:>3}{r['always_up']:>8.1%}"
                 f"{c.get('honest', float('nan')):>10.1%}{r['perfect_market_timing']:>9.1%}"
                 f"{r['perfect_relative_strength_mean']:>14.1%}{r['perfect_relative_strength_median']:>14.1%}"
                 f"{c.get('leak_fwd_ret', float('nan')):>13.1%}{c.get('leak_sign_noise_sd1', float('nan')):>9.1%}")
    L.append("  誠實模型貼著猜漲走；90% 只有「知道未來報酬本身」或訊噪比 ≥1.3 的外洩才拿得到。")
    L.append("  守門：回測準確率 > 相對強弱(中位) 或 |rank IC| > %.2f → 外洩嫌疑，CI 擋下。" % IC_CEILING)
    if flags:
        L.append("  ⚠ 外洩嫌疑：")
        for f in flags:
            L.append(f"     {f['market']} h={f['horizon']} {f['model']}: acc {f['accuracy']:.1%} "
                     f"IC {f['rank_ic']:+.3f} —— {f['reason']}")
    else:
        L.append("  既有回測結果對照：沒有煙。")
    return "\n".join(L)


def main(argv: list[str]) -> None:
    pnl = pd.read_parquet(DATA / "features/panel.parquet")
    if "market" not in pnl.columns:
        pnl["market"] = "TW"
    res_p = DATA / "backtest" / "results.parquet"
    results = pd.read_parquet(res_p) if res_p.exists() else pd.DataFrame()
    if "--guard" in argv and OUT.exists():
        saved = json.loads(OUT.read_text(encoding="utf-8"))
        flags = guard(results, saved["oracles"])
        print(report(saved["oracles"], saved.get("canaries", []), flags))
        sys.exit(1 if flags else 0)
    orc = oracles(pnl)
    can = [] if "--no-canary" in argv else canaries(pnl)
    flags = guard(results, orc)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"as_of": pnl["date"].max().strftime("%Y%m%d"),
                               "months": 12, "stride": 5, "ic_ceiling": IC_CEILING,
                               "oracles": orc, "canaries": can, "flags": flags},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(report(orc, can, flags))
    print(f"\n已存：{OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
