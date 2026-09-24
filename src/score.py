"""計分：判斷模型到底有沒有價值。

核心觀念（照重要性排序，多數人只看第一項就是問題所在）：
  1. 準確率  —— 最直覺，但單獨看幾乎無意義（台股長期偏多，猜漲就有 5 成多）。
  2. 對 baseline 的超額 —— 真正該看的。贏不過 always_up 就是沒有價值。
  3. Brier score —— 同時衡量方向與信心，越低越好。
  4. 校準曲線 —— 說 70% 的那批，是否真有 70% 上漲。
  5. 統計顯著性 —— 樣本不夠就不要宣稱有 edge。

2026-09-24 審核後改的五件事（每一件都對應一個「數字存在但定義錯」的問題）：
  a. **中性判斷不進命中率**。prob_up=0.5 且 exp_ret=0 的列（判斷檔明寫「不做方向判斷」）
     原本因 direction=sign(exp_ret) 被記成看多，首批 8/49 筆如此，命中率 65.3% 有 1.9pp
     來自 always_up 的答案。現在從**不可變欄位**推導 abstain（結算列是 append-only，
     已結算的 8 筆不能改，所以必須在讀取端推導，不能只在 settle 寫入端做）。
  b. **兩種方向並列**。`correct` 量的是決策（方向＝sign(exp_ret)，METHOD §1.1 的設計），
     Brier 量的是機率。兩者是不同的預測：momentum 的 exp_ret 被歷史均值主導，
     台股 20 日有 99.9% 的 direction 是 +1，「momentum 準確率」其實就是 always_up。
     所以另列 accuracy_by_prob（prob_up>0.5 判方向）與 inconsistency_rate，不改 correct 的定義。
  c. **Brier 技能對長期基本率**。原本對「當批事後上漲比率」算，任何常數預測必為負
     （Brier(常數 p) = clim(1−clim) + (p−clim)²）；改對 config.LONGRUN_BASE_RATE（PIT、
     事前固定），舊定義保留為 brier_skill_insample 並標明是事後氣候學。
  d. **p 值改逐日群集 t 並依重疊折減**。`_binom_p` 假設 i.i.d.，同日 50 檔＋日頻重疊下
     Monte Carlo 實測名目 α=0.05 的假陽性 0.16–0.44。逐日配對差的 t（日為群集），
     再除以 √(h/stride)，有效天數 < MIN_EFF_DAYS 時不出 p 值 —— 首批只有 1 天，
     任何方法都給不出 p 值，原本卻印 0.5–0.73。折減對「每日重抽」型的預測偏保守，
     這是憲法第四條刻意的方向。
  e. **分市場另列**。台美基本率差 8.7pp，混池的氣候學被市場間變異墊高、技能分數系統性
     灌高。by_horizon 保留（下游與測試依賴），另加 by_horizon_market[h][market]。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SETTLEMENTS, LONGRUN_BASE_RATE

MIN_EFF_DAYS = 4          # 有效天數（天數 ÷ 重疊倍數）低於此值不出 p 值


def _load() -> pd.DataFrame:
    if not SETTLEMENTS.exists():
        return pd.DataFrame()
    rows = []
    for l in SETTLEMENTS.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        try:                      # 一行壞掉不該讓整份成績單掛掉
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(rows)


def _binom_p_iid(k: int, n: int, p0: float) -> float:
    """單尾二項檢定：k 次命中 / n 次，虛無假設命中率 = p0。

    **只適用於獨立樣本。** 同日 50 檔互相相關、日頻預測在 h 日尺度上重疊，
    這兩件事讓它的假陽性率是名目值的 3–9 倍（2026-09-24 Monte Carlo）。
    成績單不再用它，保留給單元測試與「獨立樣本」的場合。
    """
    if n == 0:
        return 1.0
    z = (k / n - p0) / math.sqrt(max(p0 * (1 - p0) / n, 1e-12))
    return 0.5 * math.erfc(z / math.sqrt(2))


_binom_p = _binom_p_iid     # 舊名，相容


def is_abstain(prob_up, exp_ret) -> pd.Series:
    """中性判斷：機率恰為 0.5 且期望值為 0。從不可變欄位推導，所以對已結算的列也適用。"""
    p = pd.to_numeric(pd.Series(prob_up), errors="coerce")
    e = pd.to_numeric(pd.Series(exp_ret), errors="coerce").fillna(0.0)
    return ((p - 0.5).abs() < 1e-9) & (e.abs() < 1e-6)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """補上計分需要的衍生欄位。不改任何既有欄位。"""
    d = df.copy()
    d["prob_up"] = pd.to_numeric(d["prob_up"], errors="coerce")
    d["abstain"] = is_abstain(d["prob_up"], d.get("exp_ret", 0.0)).to_numpy()
    d["correct_eff"] = pd.to_numeric(d["correct"], errors="coerce").where(~d["abstain"])
    dir_prob = np.sign(d["prob_up"] - 0.5)                      # 恰 0.5 → 0
    d["correct_prob"] = ((dir_prob == d["actual_direction"]).astype(float)
                         .where(dir_prob != 0))
    pdir = pd.to_numeric(d["predicted_direction"], errors="coerce").fillna(0)
    d["inconsistent"] = ((dir_prob != 0) & (pdir != 0) & (dir_prob != pdir)).astype(float)
    if "market" not in d.columns:
        d["market"] = d["code"].map(lambda c: "TW" if str(c).isdigit() else "US")
    return d


def _overlap(g: pd.DataFrame, h: int) -> float:
    """重疊倍數 = h ÷ 取樣間隔。取樣間隔由 as_of 的間距推得（日頻→1）。"""
    days = sorted(pd.to_datetime(pd.Series(g["as_of"].unique()), format="%Y%m%d",
                                 errors="coerce").dropna())
    if len(days) < 2:
        return float(h)
    gaps = np.diff([d.toordinal() for d in days])
    stride = max(float(np.median(gaps)), 1.0)
    return max(h / stride, 1.0)


def cluster_t(diff_by_day: pd.Series, overlap: float) -> tuple[float | None, float]:
    """逐日群集 t，依重疊倍數折減。回傳 (t_adj 或 None, 有效天數)。"""
    x = diff_by_day.dropna()
    n_eff = len(x) / overlap
    if len(x) < 2 or x.std(ddof=1) == 0 or n_eff < MIN_EFF_DAYS:
        return None, round(n_eff, 1)
    t = float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) / math.sqrt(overlap)
    return t, round(n_eff, 1)


def _p_from_t(t: float | None, n_eff: float) -> float | None:
    """單尾 p 值（H1：模型比對照好）。自由度取有效天數 − 1。"""
    if t is None:
        return None
    from scipy.stats import t as tdist
    return round(float(tdist.sf(t, df=max(n_eff - 1, 1))), 4)


def calibration(df: pd.DataFrame, bins: int = 10) -> list[dict]:
    if df.empty:
        return []
    d = df.copy()
    d["y"] = (d["actual_direction"] > 0).astype(int)
    d["bin"] = pd.cut(d["prob_up"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = []
    for b, g in d.groupby("bin", observed=True):
        out.append({"bin": str(b), "mid": float(b.mid), "n": int(len(g)),
                    "predicted": float(g["prob_up"].mean()),
                    "actual": float(g["y"].mean())})
    return out


def _longrun_clim(g: pd.DataFrame, h: int) -> float:
    """該批列各自市場的長期基本率 Brier（r(1−r)）的平均。混池時等於按列加權。"""
    r = g["market"].map(lambda m: LONGRUN_BASE_RATE.get((m, int(h))))
    r = pd.to_numeric(r, errors="coerce")
    if r.isna().all():
        return float("nan")
    r = r.fillna(r.mean())
    return float((r * (1 - r)).mean())


def _row(name: str, g: pd.DataFrame, base: pd.DataFrame, h: int,
         clim: float, version_split: bool = False) -> dict:
    """一列成績。**所有列（彙總、分版本、分市場）必須是同一個 schema** ——
    少一個鍵，下游就會 KeyError，而那要等到有結算資料才會炸。"""
    scored = g[g["correct_eff"].notna()]
    n, n_s = len(g), len(scored)
    k = int(scored["correct_eff"].sum())
    acc = k / n_s if n_s else float("nan")
    brier = float(g["brier"].mean())
    brier_clim_in = clim * (1 - clim)
    brier_clim_lr = _longrun_clim(g, h)
    cov = (float(g["in_interval"].dropna().mean())
           if "in_interval" in g and g["in_interval"].notna().any() else None)
    iscore = (float(g["interval_score"].dropna().mean())
              if "interval_score" in g and g["interval_score"].notna().any() else None)
    cp = g["correct_prob"].dropna()
    acc_prob = float(cp.mean()) if len(cp) else None
    inc = g.loc[~g["abstain"], "inconsistent"]
    inc_rate = float(inc.mean()) if len(inc) else None

    # 與 always_up 相同 (as_of, code) 配對比較，避免樣本不同造成假差距。
    # 配對前先剔除 abstain 列：中性列沒有方向，配上去等於拿對照組的答案灌水。
    paired, t_adj, n_eff, p_val, n_days = None, None, 0.0, None, int(g["as_of"].nunique())
    if len(base) and name.split("@")[0] != "always_up" and n_s:
        j = scored.merge(base[["as_of", "code", "correct"]], on=["as_of", "code"],
                         suffixes=("", "_base"))
        if len(j):
            j["d"] = j["correct_eff"] - pd.to_numeric(j["correct_base"], errors="coerce")
            paired = float(j["d"].mean())
            by_day = j.groupby("as_of")["d"].mean()
            t_adj, n_eff = cluster_t(by_day, _overlap(j, h))
            p_val = _p_from_t(t_adj, n_eff)
    else:
        n_eff = round(n_days / _overlap(g, h), 1) if n_days else 0.0
    return {
        "model": name, "family": g["model_family"].iloc[0], "n": n,
        "n_scored": n_s, "n_abstain": int(n - n_s),
        "accuracy": round(acc, 4) if n_s else None,
        # prob_up>0.5 判方向的準確率；與 accuracy 差距大＝方向與機率是兩個模型
        "accuracy_by_prob": None if acc_prob is None else round(acc_prob, 4),
        "inconsistency_rate": None if inc_rate is None else round(inc_rate, 4),
        "brier": round(brier, 4),
        # 對長期基本率（事前常數）的技能；>0 才代表比基本率多知道一點什麼
        "brier_skill": (round(1 - brier / brier_clim_lr, 4)
                        if brier_clim_lr == brier_clim_lr and brier_clim_lr > 0 else None),
        # 對當批事後上漲比率的技能（舊定義）。任何常數預測必為負，只供對照。
        "brier_skill_insample": (round(1 - brier / brier_clim_in, 4) if brier_clim_in > 0 else None),
        # 名目 80% 區間：覆蓋率要看，但只看它會獎勵開得過寬的區間，
        # 所以並列 interval_score（含寬度罰項，越低越好）。
        "coverage_80": None if cov is None else round(cov, 4),
        "interval_score": None if iscore is None else round(iscore, 5),
        "mean_prob": round(float(g["prob_up"].mean()), 4),
        "edge_vs_always_up": None if paired is None else round(paired, 4),
        "edge_t_adj": None if t_adj is None else round(t_adj, 2),
        "n_days": n_days,
        "n_eff_days": n_eff,
        # 逐日配對差的單尾 p（日群集、重疊折減）；有效天數不足 → None，不是 0.5
        "p_value_vs_base": p_val,
        "avg_return_when_long": round(float(
            g[g["predicted_direction"] == 1]["actual_return"].mean()), 5)
            if (g["predicted_direction"] == 1).any() else None,
        "calibration": calibration(g),
        "version_split": version_split,
    }


def _block(gh: pd.DataFrame, h: int) -> dict:
    base = gh[gh["model"] == "always_up"]
    clim = float((gh["actual_direction"] > 0).mean())
    models = [_row(m, g, base, h, clim) for m, g in gh.groupby("model")]
    # 同一個模型名底下若有多個版本，各版本另外分列。
    # 不分列的話，改過方法的模型會與舊版混在同一個平均裡，
    # 而那個平均不對應任何一套實際跑過的方法。
    if "model_version" in gh.columns:
        for m, g in gh.groupby("model"):
            vs = sorted(v for v in g["model_version"].dropna().unique() if v)
            if len(vs) < 2:
                continue
            models += [_row(f"{m}@{v}", g[g["model_version"] == v], base, h, clim, True)
                       for v in vs]
    models.sort(key=lambda x: x["brier"])
    lr = {mk: LONGRUN_BASE_RATE.get((mk, int(h))) for mk in sorted(gh["market"].unique())}
    return {
        "base_rate_up": round(clim, 4),
        "brier_climatology": round(clim * (1 - clim), 4),
        "longrun_base_rate": lr,
        "n_days": int(gh["as_of"].nunique()),
        "models": models,
    }


def summary() -> dict:
    df = _load()
    if df.empty:
        return {"status": "no_settlements", "rows": 0}
    df = prepare(df)
    res = {"rows": int(len(df)),
           "date_range": [df["as_of"].min(), df["target_date"].max()],
           "by_horizon": {}, "by_horizon_market": {}}
    for h, gh in df.groupby("horizon"):
        res["by_horizon"][str(h)] = _block(gh, int(h))
        res["by_horizon_market"][str(h)] = {
            str(mk): _block(gm, int(h)) for mk, gm in gh.groupby("market")}
    return res


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=2))
