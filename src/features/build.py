"""Point-in-time 特徵工程。

鐵則：build(panel, as_of) 產生的每一個特徵值，都只能用到 as_of 當日
（含）以前的資料。函式第一行就強制截斷，讓「未來函數」在結構上不可能發生。
標籤（y_*）僅在訓練時另行計算，且訓練資料的 as_of 必須早到標籤已實現。
"""
from __future__ import annotations
import hashlib, json
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret_1", "ret_5", "ret_20", "ret_60", "vol_20", "ma_gap_20", "ma_gap_60",
    "rsi_14", "vol_ratio", "turnover_20", "foreign_5", "foreign_20", "trust_5",
    "xs_ret_20", "mkt_ret_5", "mkt_ret_20", "dist_high_60",
    "margin_chg_5", "margin_chg_20", "short_ratio",
]

def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def _compute(df: pd.DataFrame) -> pd.DataFrame:
    """特徵計算本體。全部使用 rolling／shift，天生只看過去。"""
    df = df.sort_values(["code", "date"]).copy()
    g = df.groupby("code", sort=False)

    df["ret_1"] = g["close"].pct_change(1)
    df["ret_5"] = g["close"].pct_change(5)
    df["ret_20"] = g["close"].pct_change(20)
    df["ret_60"] = g["close"].pct_change(60)
    df["vol_20"] = g["ret_1"].transform(lambda s: s.rolling(20).std())
    df["ma_gap_20"] = df["close"] / g["close"].transform(lambda s: s.rolling(20).mean()) - 1
    df["ma_gap_60"] = df["close"] / g["close"].transform(lambda s: s.rolling(60).mean()) - 1
    df["rsi_14"] = g["close"].transform(_rsi)
    df["vol_ratio"] = (g["volume"].transform(lambda s: s.rolling(5).mean())
                       / g["volume"].transform(lambda s: s.rolling(20).mean()))
    df["turnover_20"] = g["value"].transform(lambda s: s.rolling(20).mean()) / 1e9
    df["dist_high_60"] = df["close"] / g["close"].transform(lambda s: s.rolling(60).max()) - 1

    # 法人買賣超：以近 20 日均量標準化，跨股可比
    avg_vol = g["volume"].transform(lambda s: s.rolling(20).mean())
    for col, win, name in (("foreign", 5, "foreign_5"), ("foreign", 20, "foreign_20"),
                           ("trust", 5, "trust_5")):
        if col in df.columns:
            df[name] = g[col].transform(lambda s, w=win: s.rolling(w).sum()) / (avg_vol * win)
        else:
            df[name] = np.nan

    # 融資券：台股特有的散戶槓桿訊號。融資急增常是短線過熱的反向指標。
    if "margin_bal" in df.columns:
        df["margin_chg_5"] = g["margin_bal"].transform(lambda s: s.pct_change(5))
        df["margin_chg_20"] = g["margin_bal"].transform(lambda s: s.pct_change(20))
        df["short_ratio"] = df["short_bal"] / df["margin_bal"].replace(0, np.nan)
    else:
        df["margin_chg_5"] = df["margin_chg_20"] = df["short_ratio"] = np.nan

    # 市場（universe 等權）報酬 —— 同樣只由截斷後的資料算出
    mkt = df.groupby("date")["ret_1"].mean().rename("mkt_ret_1")
    mkt_df = mkt.to_frame()
    mkt_df["mkt_ret_5"] = mkt_df["mkt_ret_1"].rolling(5).sum()
    mkt_df["mkt_ret_20"] = mkt_df["mkt_ret_1"].rolling(20).sum()
    df = df.merge(mkt_df[["mkt_ret_5", "mkt_ret_20"]], left_on="date", right_index=True, how="left")

    # 橫斷面相對強度：同日跨股排名，仍只用當日資料
    df["xs_ret_20"] = df.groupby("date")["ret_20"].rank(pct=True)
    df["as_of"] = df["date"]
    out = df[["as_of", "code", "close"] + FEATURE_COLS].copy()
    # 除以零會產生 inf（例如融資餘額由 0 起算的 pct_change、或零成交量日）。
    # sklearn 會直接拋錯，而 inf 混進訓練集比缺值更危險 —— 一律轉成缺值，
    # 交給 imputer 以中位數填補。
    out[FEATURE_COLS] = out[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
    return out


def build(panel: pd.DataFrame, as_of: pd.Timestamp | str) -> pd.DataFrame:
    """安全路徑：先截斷到 as_of，再算特徵。正式預測一律走這條。"""
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] <= as_of]          # ← 前視偏誤的唯一防線
    if df.empty:
        return pd.DataFrame()
    out = _compute(df)
    out = out[out["as_of"] == out["as_of"].max()]
    return out.reset_index(drop=True)


def build_all(panel: pd.DataFrame) -> pd.DataFrame:
    """快速路徑：一次算完所有日期，供訓練與回測使用。

    之所以安全：所有特徵都由 rolling/shift/同日橫斷面構成，本質上只看
    當日與過去。tests/test_no_lookahead.py 會逐日比對這條路徑與 build()
    的結果是否完全相同 —— 一旦有人不小心引入未來資料，測試就會失敗。
    """
    if panel.empty:
        return pd.DataFrame()
    return _compute(panel).reset_index(drop=True)

def labels(panel: pd.DataFrame, as_of, horizon: int) -> pd.DataFrame:
    """未來 horizon 交易日的實際報酬（僅供訓練／結算，禁止進入特徵）。"""
    as_of = pd.Timestamp(as_of)
    out = []
    for code, sub in panel.sort_values("date").groupby("code", sort=False):
        sub = sub.reset_index(drop=True)
        idx = sub.index[sub["date"] == as_of]
        if len(idx) == 0: continue
        i = idx[0]
        j = i + horizon
        if j >= len(sub): continue
        p0, p1 = sub.loc[i, "close"], sub.loc[j, "close"]
        if not (p0 > 0 and p1 > 0): continue
        out.append({"as_of": as_of, "code": code, "horizon": horizon,
                    "target_date": sub.loc[j, "date"], "price_start": p0,
                    "price_end": p1, "fwd_ret": p1 / p0 - 1,
                    "y": int(p1 > p0)})
    return pd.DataFrame(out)

def feature_hash(row: pd.Series) -> str:
    """特徵指紋：寫進預測記錄，日後可驗證預測當下真的用了這些輸入。"""
    payload = {c: (None if pd.isna(row.get(c)) else round(float(row[c]), 8))
               for c in FEATURE_COLS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def labels_all(panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """向量化版標籤（供訓練／回測）。語意必須與 labels() 完全相同。"""
    p = panel.sort_values(["code", "date"]).copy()
    g = p.groupby("code", sort=False)
    p["price_end"] = g["close"].shift(-horizon)
    p["target_date"] = g["date"].shift(-horizon)
    p = p.dropna(subset=["price_end", "target_date"])
    p = p[(p["close"] > 0) & (p["price_end"] > 0)]
    p["fwd_ret"] = p["price_end"] / p["close"] - 1
    p["y"] = (p["fwd_ret"] > 0).astype(int)
    p["horizon"] = horizon
    return p.rename(columns={"date": "as_of", "close": "price_start"})[
        ["as_of", "code", "horizon", "target_date", "price_start",
         "price_end", "fwd_ret", "y"]].reset_index(drop=True)
