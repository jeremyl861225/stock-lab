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
