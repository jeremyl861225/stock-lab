"""把 FinMind 快取組成 tidy panel（date × code）並存 parquet。

法人資料在來源端是 long 格式（每個法人別一列、buy/sell 分開），
必須自己 pivot 成淨額；融資券則提供台股特有的散戶槓桿訊號。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW, FEATURES

INST_MAP = {"Foreign_Investor": "foreign", "Investment_Trust": "trust",
            "Dealer_self": "dealer_self", "Dealer_Hedging": "dealer_hedge",
            "Foreign_Dealer_Self": "foreign_dealer"}


def _read(kind: str) -> pd.DataFrame:
    d = RAW / "finmind" / kind
    rows = []
    for f in sorted(d.glob("*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))["payload"]
    return pd.DataFrame(rows)


def adjust_for_corporate_actions(px: pd.DataFrame) -> pd.DataFrame:
    """除權息還原（backward adjustment）。

    不做這件事，所有跨越除權息日的報酬都是錯的，而且錯得離譜：
    緯穎 2026-09-02 配股後帳面「單日 -66.5%」—— 台股跌停才 10%，
    這個假跌幅會讓動能特徵、標籤、乃至整個模型全部中毒。

    FinMind 的 TaiwanStockDividendResult 直接給除權前後參考價，
    比自行偵測跳空精確（連 1% 的小額配息都還原得到）。
    調整因子 = after_price / before_price，對除權日「之前」的價格連乘。
    """
    div = _read("div")
    if div.empty:
        return px
    div = div.rename(columns={"stock_id": "code"})
    div["date"] = pd.to_datetime(div["date"])
    for c in ("before_price", "after_price"):
        div[c] = pd.to_numeric(div[c], errors="coerce")
    div = div[(div["before_price"] > 0) & (div["after_price"] > 0)]
    if div.empty:
        return px

    px = px.sort_values(["code", "date"]).copy()
    px["adj_factor"] = 1.0
    for code, events in div.groupby("code"):
        mask_code = px["code"] == code
        if not mask_code.any():
            continue
        factor = pd.Series(1.0, index=px.index[mask_code])
        dates = px.loc[mask_code, "date"]
        for e in events.itertuples():
            f = e.after_price / e.before_price
            if not (0.05 < f < 1.5):      # 異常因子不套用，寧可不調整也不要弄壞資料
                continue
            factor[dates < e.date] *= f   # 只調整除權日之前
        px.loc[mask_code, "adj_factor"] = factor.values

    for c in ("open", "high", "low", "close"):
        if c in px.columns:
            px[c] = px[c] * px["adj_factor"]
    return px


def repair_unexplained_splits(px: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """處理除權息資料漏掉的公司行動。分兩種情況，處理方式刻意不同。

    門檻為何是 30% 而非 11%：
      台股漲跌停 ±10%，但新上市／興櫃轉上市股確實有 11~30% 的真實波動
      （鴻勁 2024-11 上市後一年內 10 次，分散且無對應公司行動）。
      把真實波動「修正」掉是製造假資料，比留著更糟。

    >30% 的跳空再分兩類 —— 這一步是我後來才想通的，第一版全部當成分割是錯的：
      (a) 比例接近 1/n（n=2..10）→ 幾乎確定是股票分割，比例還原。
          例：國巨 2025-08-25 因子 0.2619 ≈ 1/4，還原後最大單日變動回到 10%。
      (b) 比例不接近任何簡單分數 → 成因不明，可能是資料錯誤，也可能是
          無漲跌幅限制期間的真實暴跌（鴻勁 2025-04-07 因子 0.6648，
          當天正是全球關稅股災）。這種情況「猜一個因子去調整」等於偽造資料，
          正確做法是截斷該日之前的歷史：寧可資料少，不要資料錯。
    """
    px = px.sort_values(["code", "date"]).copy()
    fixed, truncated = [], []
    drop_idx = []
    for code, g in px.groupby("code", sort=False):
        chg = g["close"].pct_change()
        events = list(g.loc[chg.abs() > threshold, "date"])
        if not events:
            continue
        mask = px["code"] == code
        factor = pd.Series(1.0, index=px.index[mask])
        dates = px.loc[mask, "date"]
        for d in events:
            pos = g.index.get_loc(g.index[g["date"] == d][0])
            if pos == 0:
                continue
            f = g["close"].iloc[pos] / g["close"].iloc[pos - 1]
            n = round(1 / f) if f > 0 else 0
            is_split = 2 <= n <= 10 and abs(f - 1 / n) < 0.03
            if is_split:
                factor[dates < d] *= f
                fixed.append((code, str(pd.Timestamp(d).date()), f"1:{n}", round(f, 4)))
            else:
                drop_idx += list(px.index[mask & (px["date"] < d)])
                truncated.append((code, str(pd.Timestamp(d).date()), round(f, 4)))
        for c in ("open", "high", "low", "close"):
            px.loc[mask, c] = px.loc[mask, c] * factor.values
    if fixed:
        print(f"  補上未記錄的股票分割 {len(fixed)} 筆：{fixed}")
    if truncated:
        print(f"  成因不明的跳空 {len(truncated)} 筆 → 截斷其之前的歷史"
              f"（共 {len(set(drop_idx))} 列）：{truncated}")
        px = px.drop(index=set(drop_idx))
    return px


def build(codes: set[str] | None = None) -> pd.DataFrame:
    px = _read("price")
    if px.empty:
        raise RuntimeError("panel 為空：請先執行 FinMind 回補")
    px = px.rename(columns={"stock_id": "code", "max": "high", "min": "low",
                            "Trading_Volume": "volume", "Trading_money": "value"})
    px = px[["date", "code", "open", "high", "low", "close", "volume", "value"]]

    inst = _read("inst")
    if not inst.empty:
        inst["net"] = inst["buy"] - inst["sell"]
        inst["who"] = inst["name"].map(INST_MAP).fillna("other")
        inst = (inst.pivot_table(index=["date", "stock_id"], columns="who",
                                 values="net", aggfunc="sum")
                    .reset_index().rename(columns={"stock_id": "code"}))
        inst["dealer"] = inst.get("dealer_self", 0).fillna(0) + inst.get("dealer_hedge", 0).fillna(0)
        keep = ["date", "code", "foreign", "trust", "dealer"]
        inst = inst[[c for c in keep if c in inst.columns]]
        px = px.merge(inst, on=["date", "code"], how="left")

    mg = _read("margin")
    if not mg.empty:
        mg = mg.rename(columns={"stock_id": "code",
                                "MarginPurchaseTodayBalance": "margin_bal",
                                "ShortSaleTodayBalance": "short_bal"})
        px = px.merge(mg[["date", "code", "margin_bal", "short_bal"]],
                      on=["date", "code"], how="left")

    if codes:
        px = px[px["code"].isin(codes)]
    px["date"] = pd.to_datetime(px["date"])
    for c in ("open", "high", "low", "close", "volume", "value",
              "foreign", "trust", "dealer", "margin_bal", "short_bal"):
        if c in px.columns:
            px[c] = pd.to_numeric(px[c], errors="coerce")
    px = (px[px["close"] > 0].sort_values(["code", "date"])
            .drop_duplicates(["code", "date"]).reset_index(drop=True))
    px = adjust_for_corporate_actions(px)
    px = repair_unexplained_splits(px)
    FEATURES.mkdir(parents=True, exist_ok=True)
    px.to_parquet(FEATURES / "panel.parquet", index=False)
    return px


if __name__ == "__main__":
    from collect import universe
    uni = universe.load()
    codes = {c["code"] for c in uni["constituents"]}
    df = build(codes)
    print(f"panel: {len(df):,} 列 × {df['code'].nunique()} 檔  "
          f"{df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"欄位: {list(df.columns)}")
    miss = df[["foreign", "trust", "margin_bal"]].isna().mean() if "foreign" in df else None
    if miss is not None:
        print("缺值率:", {k: f"{v:.1%}" for k, v in miss.items()})
    print(df[df.code == "2330"].tail(3).to_string(index=False))
