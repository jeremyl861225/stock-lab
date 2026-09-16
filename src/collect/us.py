# -*- coding: utf-8 -*-
"""美股資料收集（yfinance）。

與台股的差異，這些會直接影響特徵可用性：
  - 美股沒有台股那種每日三大法人買賣超與融資餘額公告，
    籌碼面特徵（foreign/trust/margin）一律缺值，由 imputer 以中位數填補。
  - yfinance 的 history(auto_adjust=True) 已還原分割與股息，不需自行處理除權息。
  - 交易日曆與台股不同，因此 panel 以 market 欄位區隔，
    市場報酬（mkt_ret）必須分市場計算，否則會把兩個市場的漲跌混在一起。
"""
from __future__ import annotations
import datetime as dt, json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW, CONFIG

# 使用者指定的 ETF ＋ 自行以市值排序取得的前十大個股
ETFS = ["BTCO", "QQQ", "VOO"]
# 取市值前 50 需要足夠寬的候選池；以 S&P500 大型股為母體，
# 實際排名仍由 yfinance 的即時市值決定，不依賴二手清單。
CANDIDATES = [
    "NVDA","AAPL","MSFT","GOOGL","AMZN","META","AVGO","TSLA","BRK-B","LLY",
    "TSM","WMT","JPM","V","ORCL","MA","XOM","COST","UNH","NFLX",
    "PG","JNJ","HD","ABBV","BAC","CRM","KO","CVX","AMD","PM",
    "TMUS","CSCO","WFC","MCD","ABT","IBM","GE","LIN","CAT","MRK",
    "NOW","PEP","ISRG","AXP","MS","GS","VZ","DIS","RTX","INTU",
    "T","AMGN","TXN","BKNG","QCOM","SPGI","PLTR","BLK","SCHW","C",
    "LOW","ADBE","HON","NEE","UBER","ETN","PGR","TJX","BSX","SYK",
    "LMT","MU","ADP","VRTX","PANW","MDT","GILD","ANET","KKR","DE",
]


def build_universe(top_n: int = 50) -> dict:
    """自行抓市值排序，不依賴二手清單 —— 與台股 universe 同一套邏輯。"""
    import yfinance as yf
    rows = []
    for t in CANDIDATES:
        try:
            i = yf.Ticker(t).info
            if i.get("marketCap"):
                rows.append({"code": t, "name": (i.get("shortName") or t)[:32],
                             "industry": (i.get("sector") or "")[:20],
                             "mcap": float(i["marketCap"])})
        except Exception:  # noqa: BLE001
            continue
        time.sleep(0.1)
    rows.sort(key=lambda r: r["mcap"], reverse=True)
    top = rows[:top_n]

    for t in ETFS:                       # ETF 無市值可比，固定納入
        try:
            import yfinance as yf
            i = yf.Ticker(t).info
            top.append({"code": t, "name": (i.get("shortName") or t)[:32],
                        "industry": "ETF", "mcap": float(i.get("totalAssets") or 0)})
        except Exception:  # noqa: BLE001
            top.append({"code": t, "name": t, "industry": "ETF", "mcap": 0.0})

    total = sum(r["mcap"] for r in top) or 1.0
    uni = {"as_of": dt.date.today().strftime("%Y%m%d"),
           "built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
           "market": "US", "method": "market_cap_top_n_plus_designated_etfs",
           "size": len(top),
           "constituents": [{**r, "weight": r["mcap"] / total} for r in top]}
    out = CONFIG / "universe_us"
    out.mkdir(exist_ok=True)
    (out / f"{uni['as_of']}.json").write_text(
        json.dumps(uni, ensure_ascii=False, indent=2), encoding="utf-8")
    (CONFIG / "universe_us_latest.json").write_text(
        json.dumps(uni, ensure_ascii=False, indent=2), encoding="utf-8")
    return uni


def load(as_of: str | None = None) -> dict:
    p = CONFIG / "universe_us_latest.json"
    if not p.exists():
        raise FileNotFoundError("尚未建立美股 universe")
    return json.loads(p.read_text(encoding="utf-8"))


def fetch_prices(codes: list[str], years: float = 2.1) -> pd.DataFrame:
    """auto_adjust=True：分割與股息已還原，不需自行處理公司行動。"""
    import yfinance as yf
    end = dt.date.today()
    start = end - dt.timedelta(days=int(365 * years))
    df = yf.download(codes, start=start.isoformat(), end=end.isoformat(),
                     auto_adjust=True, progress=False, group_by="ticker",
                     threads=True)
    rows = []
    for c in codes:
        try:
            g = df[c].dropna(subset=["Close"]) if len(codes) > 1 else df.dropna(subset=["Close"])
        except KeyError:
            continue
        for d, r in g.iterrows():
            rows.append({"date": pd.Timestamp(d).tz_localize(None), "code": c,
                         "open": r["Open"], "high": r["High"], "low": r["Low"],
                         "close": r["Close"], "volume": r["Volume"],
                         "value": r["Close"] * r["Volume"]})
    out = pd.DataFrame(rows)
    d = RAW / "us"
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "prices.parquet", index=False)
    return out


def fetch_fundamentals(codes: list[str]) -> pd.DataFrame:
    import yfinance as yf
    rows = []
    for c in codes:
        try:
            i = yf.Ticker(c).info
            rows.append({
                "code": c, "PER": i.get("trailingPE"), "PBR": i.get("priceToBook"),
                "dividend_yield": (i.get("dividendYield") or 0),
                "rev_yoy": i.get("revenueGrowth"),
                "rev_yoy3m": i.get("revenueGrowth"),
                "rev_mom": None, "rev_month": "季報",
                "shortPct": i.get("shortPercentOfFloat"),
                "targetMean": i.get("targetMeanPrice"),
            })
        except Exception:  # noqa: BLE001
            rows.append({"code": c})
        time.sleep(0.1)
    out = pd.DataFrame(rows)
    (RAW / "us").mkdir(parents=True, exist_ok=True)
    out.to_parquet(RAW / "us/fundamentals.parquet", index=False)
    return out


if __name__ == "__main__":
    uni = build_universe(50)
    codes = [c["code"] for c in uni["constituents"]]
    print(f"美股 universe：{len(codes)} 檔 → {', '.join(codes)}")
    px = fetch_prices(codes)
    print(f"價量：{len(px):,} 列，{px['date'].min().date()} ~ {px['date'].max().date()}")
    fd = fetch_fundamentals(codes)
    print(f"基本面：{fd['PER'].notna().sum()}/{len(fd)} 檔有 PER、"
          f"{fd['rev_yoy'].notna().sum()} 檔有營收成長")
