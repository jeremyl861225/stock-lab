"""笨基準線 —— 判斷其他模型有無價值的唯一標尺。

每條線都必須同時給方向機率與期望報酬，否則無法與真實模型並排比較。
`always_up` 的期望報酬用「無條件歷史平均」，這是最誠實的笨答案：
不看任何特徵，就賭市場長期往上。要打敗它，模型得證明自己真的看懂了什麼。

**基準線一旦上線就凍結**（METHOD.md §2）。要改就當成新模型另開一個名字，
否則之前所有的比較都不可比。`mech_core` 是依這條規則新增的，不是改掉舊的。
"""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd

from models.quantiles import quantiles

VERSION = "1.3.0"     # 1.3.0：新增 mech_core（既有三條線未變動）


def _hist_mean_ret(panel: pd.DataFrame, as_of, horizon: int) -> float:
    """as_of 之前、已實現的 horizon 期報酬平均。不得使用未來資料。"""
    hist = panel[panel["date"] <= pd.Timestamp(as_of)]
    dates = sorted(hist["date"].unique())
    if len(dates) < horizon + 2:
        return 0.0
    cutoff = dates[-(horizon + 1)]
    h = hist.sort_values(["code", "date"]).copy()
    h["fwd"] = h.groupby("code", sort=False)["close"].shift(-horizon) / h["close"] - 1
    v = h[(h["date"] <= cutoff)]["fwd"].dropna()
    return float(v.mean()) if len(v) else 0.0


def _hist_up_rate(panel: pd.DataFrame, as_of, horizon: int) -> float:
    """as_of 之前、已實現的 horizon 期上漲比率。與 _hist_mean_ret 同一條截斷規則。"""
    hist = panel[panel["date"] <= pd.Timestamp(as_of)]
    dates = sorted(hist["date"].unique())
    if len(dates) < horizon + 2:
        return 0.55
    cutoff = dates[-(horizon + 1)]
    h = hist.sort_values(["code", "date"]).copy()
    h["fwd"] = h.groupby("code", sort=False)["close"].shift(-horizon) / h["close"] - 1
    v = h[h["date"] <= cutoff]["fwd"].dropna()
    return float((v > 0).mean()) if len(v) else 0.55


def always_up(feats, horizon, as_of, panel) -> pd.DataFrame:
    """永遠猜漲，幅度用無條件歷史平均。台股長期偏多，這條線不好打敗。"""
    m = _hist_mean_ret(panel, as_of, horizon)
    n = len(feats)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": 0.55,
        "exp_ret": np.full(n, m), "ret_q10": np.full(n, m - 0.08),
        "ret_q90": np.full(n, m + 0.08), "direction": 1,
        "rationale": [f"baseline: 無條件看多（歷史均報酬 {m:+.2%}）"] * n,
    })


def random_walk(feats, horizon, as_of, panel) -> pd.DataFrame:
    """隨機猜。seed 由 (as_of, code, horizon) 決定，結果可完全重現。"""
    probs, rets = [], []
    for c in feats["code"]:
        h = hashlib.sha256(f"{as_of}|{c}|{horizon}".encode()).digest()
        p = int.from_bytes(h[:4], "big") / 2**32
        probs.append(p)
        rets.append((p - 0.5) * 0.10)
    p = np.array(probs); r = np.array(rets)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": p, "exp_ret": r,
        "ret_q10": r - 0.08, "ret_q90": r + 0.08,
        "direction": np.where(r >= 0, 1, -1),
        "rationale": ["baseline: 隨機（seed 可重現）"] * len(p),
    })


def momentum(feats, horizon, as_of, panel) -> pd.DataFrame:
    """動能延續：過去 20 日漲就猜續漲。最便宜的真實訊號。"""
    r20 = feats["ret_20"].fillna(0.0).to_numpy()
    p = np.clip(0.5 + np.tanh(r20 * 5) * 0.12, 0.05, 0.95)
    m = _hist_mean_ret(panel, as_of, horizon)
    exp = m + np.tanh(r20 * 4) * 0.03
    vol = feats["vol_20"].fillna(0.02).to_numpy() * np.sqrt(horizon)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": p, "exp_ret": exp,
        "ret_q10": exp - 1.2816 * vol, "ret_q90": exp + 1.2816 * vol,
        "direction": np.where(exp >= 0, 1, -1),
        "rationale": [f"baseline: 20日動能 {x:+.1%}" for x in r20],
    })


def _xs_rank(v: pd.Series, groups: pd.Series | None) -> pd.Series:
    """當日該市場內的百分位排名，置中到 [−0.5, +0.5]。缺值給 0（中性）。

    必須**分市場**算：台美的交易日與漲跌互不相干，
    把兩地擠進同一個橫斷面排名，等於拿台股的跌幅去決定美股的名次。
    """
    if groups is None:
        r = v.rank(pct=True) - 0.5
    else:
        r = v.groupby(groups).rank(pct=True) - 0.5
    return r.fillna(0.0)


def mech_core(feats, horizon, as_of, panel) -> pd.DataFrame:
    """手寫判斷的機械核心：20 日反轉 × 月營收年增。

    **為什麼要有這條線**（2026-09-19 實測）：把 2026-09-18 的台股 50 檔手寫 p20
    對簡報欄位做 Spearman，得到 ret_20 −0.62、rsi_14 −0.59、dist_high_60 −0.52、
    rev_yoy +0.62；六個特徵的 rank-R² 是 0.66，而與 momentum 基準線的相關是 −0.62。
    也就是說手寫判斷有三分之二可以被「跌深 ＋ 營收強」這兩句話複製，
    而且它與最便宜的既有基準線剛好是鏡像 —— 兩個鏡像模型裡必定有一個看起來很準，
    那不是證據。要證明judgment有增量價值，它必須贏過自己的機械核心，不是贏過猜漲。

    規格（**是規格不是擬合**，權重由我指定並自今日凍結）：
        tilt = 0.5 × 反轉分位 + 0.5 × 營收分位      （美股無月營收 → 只用反轉）
        p    = clip(歷史基本率 + tilt × SPREAD, 0.30, 0.70)
        幅度 = 0.85 × vol_20 × √h，不做偏度主張（up = −dn）
        分位數走對數常態，z = 1.2816（機械模型用精確分位，1.36 是判斷專用）

    SPREAD = 0.12 讓 p 的橫斷面標準差約 0.035，與手寫判斷的 0.027 同量級 ——
    這只影響 Brier 與準確率，不影響 rank IC（等級相關對尺度不變）。

    ⚠️ **這條線的兩半，歷史證據強弱相反**（2026-09-19 實測，PIT 宇宙、511 個交易日、
    重疊調整後的 t）：

        訊號            h=5            h=20           h=60
        營收年增   IC +0.069 t +2.69   +0.129 t +2.42  +0.188 t +2.32
        反轉       IC −0.045 t −1.65   −0.074 t −1.54  −0.042 t −0.46

    營收年增是唯一 |t| > 2 的；**反轉那一半的 IC 在三個期間全部是負的** ——
    也就是這段樣本裡動能有效、反轉無效（但 |t| ≈ 1.5，不足以宣稱反轉是反指標，
    只能說「反轉沒有被證實」）。
    刻意仍以 0.5/0.5 保留反轉：這條線的工作是**複製手寫判斷在做的事**
    （2026-09-18 台股手寫 p20 與 ret_20 的 Spearman 是 −0.62，確實在賭反轉），
    不是做一條最好的線。把它改成只用營收，就不再是那面鏡子了。
    比較的時候要記得：判斷若輸給這條線，有可能是輸在反轉那一半。
    """
    n = len(feats)
    mk = feats["market"] if "market" in feats.columns else None
    rev_rank = _xs_rank(feats["ret_20"], mk)          # 漲多的排名高
    tilt = -rev_rank                                   # 反轉：跌深的給正分
    src = "20日反轉"
    if "rev_yoy" in feats.columns and feats["rev_yoy"].notna().any():
        # 該市場完全沒有月營收（美股）時不要把它算進去 ——
        # 全缺值經 fillna(0) 會變成一整欄中性分，等於偷偷把權重折半。
        has = (feats["rev_yoy"].notna().groupby(mk).transform("any")
               if mk is not None else pd.Series(True, index=feats.index))
        yoy = _xs_rank(feats["rev_yoy"], mk)
        tilt = np.where(has, 0.5 * tilt + 0.5 * yoy, tilt)
        src = "20日反轉×營收年增（無月營收者僅用反轉）"
    # 基本率也分市場。台股兩年等權 +272%、美股不同，共用一個基本率等於
    # 拿台股的多頭去墊高美股的機率（既有三條線是共用的，但它們已凍結，不動）。
    if mk is not None and "market" in panel.columns:
        rates = {m: _hist_up_rate(panel[panel["market"] == m], as_of, horizon)
                 for m in mk.unique()}
        base_v = mk.map(rates).to_numpy(dtype=float)
        base = float(np.mean(list(rates.values())))
    else:
        base = _hist_up_rate(panel, as_of, horizon)
        base_v = np.full(n, base)
    p = np.clip(base_v + np.asarray(tilt) * 0.12, 0.30, 0.70)

    vol = feats["vol_20"].fillna(feats["vol_20"].median()).fillna(0.02).to_numpy()
    mag = 0.85 * vol * np.sqrt(horizon)
    exp = mag * (2 * p - 1)                            # skew=0：不做不對稱主張
    sig = vol * np.sqrt(horizon)
    q10, q90 = quantiles(exp, sig, z=1.2816)
    return pd.DataFrame({
        "code": feats["code"].values, "prob_up": p, "exp_ret": exp,
        "ret_q10": q10, "ret_q90": q90,
        "direction": np.where(exp >= 0, 1, -1),
        "rationale": [f"baseline: {src}（基本率 {b:.1%}）" for b in base_v],
    })


ALL = {"always_up": always_up, "random": random_walk, "momentum": momentum,
       "mech_core": mech_core}
