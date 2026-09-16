"""把四個面向的資料整合成一份分析簡報，供判斷使用。

月營收特別處理 point-in-time：用 create_time（公告日）而非 revenue_month，
否則會用到當時還沒公布的營收 —— 這是基本面資料最常見的未來函數。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW, DATA
from features.build import build
from collect import universe


def _read(kind: str) -> pd.DataFrame:
    rows = []
    for f in sorted((RAW / "finmind" / kind).glob("*.json")):
        rows += json.loads(f.read_text(encoding="utf-8"))["payload"]
    return pd.DataFrame(rows)


def fundamentals(as_of: str) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of)
    per = _read("per")
    if not per.empty:
        per["date"] = pd.to_datetime(per["date"])
        per = (per[per["date"] <= as_of_ts].sort_values("date")
               .groupby("stock_id").tail(1)
               .rename(columns={"stock_id": "code"})[["code", "PER", "PBR", "dividend_yield"]])

    rev = _read("rev")
    out = []
    if not rev.empty:
        # FinMind 只對近期資料填 create_time，舊資料是空字串。
        # 缺失者用法定公告期限推定：台股月營收須於次月 10 日前公告。
        # 不補這一步，25 筆只有 7 筆通過過濾，基本面會整片消失。
        ct = pd.to_datetime(rev["create_time"], errors="coerce")
        ym = rev["revenue_year"].astype(int) * 12 + rev["revenue_month"].astype(int)
        est = pd.to_datetime(dict(year=(ym // 12) + (ym % 12 == 0).astype(int) * 0,
                                  month=(ym % 12) + 1, day=10), errors="coerce")
        nxt = pd.to_datetime([
            f"{y + (1 if m == 12 else 0)}-{(1 if m == 12 else m + 1):02d}-10"
            for y, m in zip(rev["revenue_year"].astype(int), rev["revenue_month"].astype(int))])
        rev["pub"] = ct.fillna(pd.Series(nxt, index=rev.index))
        rev = rev[rev["pub"] <= as_of_ts]      # ← 只用已公告的
        rev["ym"] = rev["revenue_year"].astype(int) * 12 + rev["revenue_month"].astype(int)
        for code, g in rev.groupby("stock_id"):
            g = g.sort_values("ym").drop_duplicates("ym", keep="last")
            if len(g) < 13:
                continue
            idx = dict(zip(g["ym"], g["revenue"]))
            cur_ym = int(g["ym"].iloc[-1])
            cur = float(idx[cur_ym])
            prev = idx.get(cur_ym - 12)
            yoy = (cur / prev - 1) if prev else None
            l3 = [idx.get(cur_ym - i) for i in range(3)]
            p3 = [idx.get(cur_ym - 12 - i) for i in range(3)]
            yoy3 = (sum(l3) / sum(p3) - 1) if all(l3) and all(p3) and sum(p3) else None
            pm = idx.get(cur_ym - 1)
            mom = (cur / pm - 1) if pm else None
            y, m = divmod(cur_ym, 12)
            if m == 0:
                y, m = y - 1, 12
            out.append({"code": code, "rev_month": f"{y}/{m:02d}",
                        "rev_yoy": yoy, "rev_yoy3m": yoy3, "rev_mom": mom})
    revdf = pd.DataFrame(out)
    if per is None or per.empty:
        return revdf
    return per.merge(revdf, on="code", how="outer") if not revdf.empty else per


def news_titles(as_of: str) -> dict[str, list[str]]:
    d = RAW / "news" / as_of
    out = {}
    if not d.exists():
        return out
    for f in d.glob("*.json"):
        if f.stem.startswith("_"):
            continue
        out[f.stem] = [x["title"] for x in
                       json.loads(f.read_text(encoding="utf-8"))["payload"]]
    return out


def us_fundamentals() -> pd.DataFrame:
    """美股基本面（yfinance）。營收成長是季報年增，與台股的月營收年增語意不同，
    欄位共用但 rev_month 標為「季報」以免誤讀。"""
    f = RAW / "us/fundamentals.parquet"
    return pd.read_parquet(f) if f.exists() else pd.DataFrame()


def build_briefing(as_of: str | None = None) -> pd.DataFrame:
    pnl = pd.read_parquet(DATA / "features/panel.parquet")
    as_of_ts = pd.Timestamp(as_of) if as_of else pnl["date"].max()
    as_of_str = as_of_ts.strftime("%Y%m%d")
    uni = universe.load()
    names = {c["code"]: c["name"] for c in uni["constituents"]}
    inds = {c["code"]: c["industry"] for c in uni["constituents"]}
    wts = {c["code"]: c["weight"] for c in uni["constituents"]}

    # 美股 universe 併入名稱／產業對照
    try:
        from collect import us as us_mod
        uus = us_mod.load()
        names.update({c["code"]: c["name"] for c in uus["constituents"]})
        inds.update({c["code"]: c.get("industry", "") for c in uus["constituents"]})
        wts.update({c["code"]: c.get("weight", 0) for c in uus["constituents"]})
    except Exception:  # noqa: BLE001
        pass

    f = build(pnl, as_of_ts)
    f = f[f["code"].isin(names)].copy()

    fund = fundamentals(as_of_str)
    usf = us_fundamentals()
    cols = ["code", "PER", "PBR", "dividend_yield", "rev_month", "rev_yoy", "rev_yoy3m", "rev_mom"]
    parts = [d for d in (fund, usf) if not d.empty]
    if parts:
        allf = pd.concat([d.reindex(columns=cols) for d in parts], ignore_index=True)
        f = f.merge(allf.drop_duplicates("code"), on="code", how="left")

    f["名稱"] = f["code"].map(names)
    f["產業"] = f["code"].map(inds)
    f["權重"] = f["code"].map(wts)
    # 一年期用的波動窗口。vol_20 是短期狀態，拿它外推一年會錯得很離譜：
    # 實測 2330 的 vol_20 年化 18.0%、vol_60 年化 36.2%（近 20 天剛好平靜）。
    # 而預測「未來 120 日實現波動」的能力 vol_60 r=0.908 > vol_20 r=0.841。
    _p = pnl.sort_values(["code", "date"]).copy()
    _p["_r"] = _p.groupby("code")["close"].pct_change()
    f["vol_60"] = f["code"].map(
        _p.groupby("code")["_r"].apply(lambda s: s.tail(60).std()))
    if "market" not in f.columns:
        f["market"] = "TW"
    return f.sort_values(["market", "權重"], ascending=[True, False]).reset_index(drop=True)


if __name__ == "__main__":
    b = build_briefing()
    nt = news_titles(pd.Timestamp(b["as_of"].iloc[0]).strftime("%Y%m%d"))
    b.to_parquet(DATA / "briefing.parquet", index=False)
    print(f"簡報：{len(b)} 檔 × {b.shape[1]} 欄，新聞覆蓋 {len(nt)} 檔")
    print("欄位:", [c for c in b.columns if c not in ("as_of",)][:30])
