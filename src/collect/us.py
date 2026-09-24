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
from zoneinfo import ZoneInfo
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
        time.sleep(THROTTLE)
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
    """取用不晚於 as_of 的最近一份快照。as_of=None 才回傳最新的那份。

    **這個參數原本是假的** —— 簽名收下 as_of 卻永遠回傳最新的 universe，
    於是任何呼叫端以為自己做了 point-in-time，實際上拿的是今天的成分股。
    那正是回測裡生存者偏差的入口，而且它不會報錯。
    （台股那支 `universe.load` 一直是真的，兩邊介面看起來一樣但行為不同，
    這種不對稱最難發現。2026-09-19 補上。）
    """
    if as_of is None:
        p = CONFIG / "universe_us_latest.json"
        if not p.exists():
            raise FileNotFoundError("尚未建立美股 universe")
        return json.loads(p.read_text(encoding="utf-8"))
    out = CONFIG / "universe_us"
    files = sorted(out.glob("*.json")) if out.exists() else []
    key = str(as_of).replace("-", "")[:8]
    ok = [f for f in files if f.stem <= key]
    if not ok:
        raise ValueError(f"{as_of} 之前沒有可用的美股 universe 快照（會造成前視偏誤）")
    return json.loads(ok[-1].read_text(encoding="utf-8"))


def codes_ever() -> set[str]:
    """所有快照中曾入選過的代號 —— panel 必須涵蓋這整組，
    否則「後來掉出去」的股票會在特徵計算時整批消失。"""
    out = CONFIG / "universe_us"
    ever: set[str] = set()
    for f in sorted(out.glob("*.json")) if out.exists() else []:
        d = json.loads(f.read_text(encoding="utf-8"))
        ever |= {c["code"] for c in d.get("constituents", [])}
    return ever


# Yahoo 在被連打時會靜默改吐舊資料（見 fetch_prices），實測每請求間隔
# 0.8 秒就不會發生。68 檔約 55 秒，這個代價遠小於靜默短給一天。
THROTTLE = 0.8


def _history(code: str, start: str | None = None, period: str | None = None,
             tries: int = 2) -> pd.DataFrame | None:
    """逐檔取 K 棒，失敗重試。回傳 None 代表這一檔真的拿不到。"""
    import yfinance as yf
    kw = {"start": start} if start else {"period": period or "1mo"}
    for _ in range(tries):
        try:
            h = yf.Ticker(code).history(auto_adjust=True, **kw)
            if len(h):
                return h.dropna(subset=["Close"])
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)
    return None


def _to_rows(g: pd.DataFrame, code: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": [pd.Timestamp(pd.Timestamp(i).date()) for i in g.index],
        "code": code,
        "open": g["Open"].to_numpy(), "high": g["High"].to_numpy(),
        "low": g["Low"].to_numpy(), "close": g["Close"].to_numpy(),
        "volume": g["Volume"].to_numpy(),
    })
    out["value"] = out["close"] * out["volume"]
    return out


def latest_session(refs: tuple[str, ...] = ("AAPL", "MSFT", "QQQ", "NVDA", "JPM"),
                   rounds: int = 4) -> pd.Timestamp:
    """用幾檔一定有量的標的問出「最新的交易日」，當作後面的驗收基準。

    不能拿本批抓回來的最大日期當基準 —— 來源鬧脾氣的時候整批一起舊，
    那個基準會跟著錯，守門就永遠過得去（2026-09-24 就是這樣過了兩天才發現）。
    來源發舊貨時連基準也會舊，所以這裡退避重問，取歷次看過的最大值。
    """
    best: pd.Timestamp | None = None
    for r in range(rounds):
        for c in refs:
            g = _history(c, period="5d")
            if g is not None and len(g):
                d = pd.Timestamp(pd.Timestamp(g.index[-1]).date())
                best = d if best is None or d > best else best
            time.sleep(THROTTLE)
        # 基準要用紐約的今天，不是這台機器的今天 —— 機器在 JST，
        # 台北早上跑的時候紐約還在前一天的盤後，兩邊差一天。
        ny_today = dt.datetime.now(ZoneInfo("America/New_York")).date()
        if best is not None and (ny_today - best.date()).days <= 1:
            break
        time.sleep(15 * (r + 1))
    if best is None:
        raise RuntimeError("連基準標的都抓不到，來源不可用")
    return best


def fetch_prices(codes: list[str], years: float = 2.1, rounds: int = 3) -> pd.DataFrame:
    """auto_adjust=True：分割與股息已還原，不需自行處理公司行動。

    **2026-09-24 查出的來源行為**：Yahoo 被連續請求時會靜默改吐舊資料 ——
    `yf.download` 批次、以及間隔太短的 `Ticker.history`，都可能少給最後一兩天，
    而且**不報錯**。那天 68 檔一度全部只到 09-21/09-22，隔一分鐘單獨重抓
    同一檔立刻拿到 09-23。表現出來只是「美股 as_of 卡在舊日期、briefing
    只剩 21 檔」，美股判斷因此連續兩天沒做成，中間沒有任何錯誤訊息。

    對策三層：每請求間隔 `THROTTLE` 秒、拿 `latest_session()` 當外部基準
    驗收、落後的退避重抓。最後仍不合格就丟例外 —— 寧可讓 prepare 停下來，
    也不要靜默寫進一份 as_of 是假的價量檔。
    """
    long_start = (dt.date.today() - dt.timedelta(days=int(365 * years))).isoformat()
    target = latest_session()

    parts: dict[str, pd.DataFrame] = {}
    for c in codes:
        g = _history(c, start=long_start)
        if g is not None:
            parts[c] = _to_rows(g, c)
        time.sleep(THROTTLE)
    empty = [c for c in codes if c not in parts]
    if not parts:
        raise RuntimeError("美股價量一檔都沒抓到，拒絕覆寫既有檔案")

    for r in range(rounds):
        lag = [c for c, df in parts.items() if df["date"].max() < target]
        if not lag:
            break
        wait = 15 * (r + 1)
        print(f"　{len(lag)} 檔尚未到 {target.date()}，等 {wait}s 再補抓")
        time.sleep(wait)
        for c in lag:
            g = _history(c, period="1mo", tries=1)
            if g is not None:
                parts[c] = (pd.concat([parts[c], _to_rows(g, c)])
                            .drop_duplicates(subset=["date"], keep="last"))
            time.sleep(THROTTLE)

    out = pd.concat(parts.values())

    # 與既有檔合併：來源發舊貨那天，已經收到的日子不可以倒退不見。
    # （原本整份覆寫，所以來源短給一次就等於把那幾天從歷史裡刪掉。）
    prev_path = RAW / "us/prices.parquet"
    if prev_path.exists():
        prev = pd.read_parquet(prev_path)
        out = pd.concat([prev, out])          # 新抓的排後面，重複時留新的
    out = (out.drop_duplicates(subset=["date", "code"], keep="last")
              .sort_values(["code", "date"]).reset_index(drop=True))

    d = RAW / "us"
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "prices.parquet", index=False)      # 先落地，守門再擋
    if empty:
        print(f"　⚠ {len(empty)} 檔抓不到價量：{', '.join(empty)}")

    # 守門：最新一個「收盤已定案」的交易日要有夠多檔。
    # 當天尚未定案的那根（Close 為 NaN 的空殼）已在 _history 裡被 dropna 濾掉，
    # 所以 target 一定是已完成的交易日；它若只有零星幾檔，就是來源缺資料，
    # 不是停牌。這種時候 briefing 會只剩那幾檔，而且不會有人發現。
    have = out.loc[out["date"] == target, "code"].nunique()
    got_n = out["code"].nunique()
    if have < 0.8 * got_n:
        miss = sorted(set(out["code"].unique()) - set(out.loc[out["date"] == target, "code"]))
        raise RuntimeError(
            f"美股基準交易日 {target.date()} 只有 {have}/{got_n} 檔有 K 棒 —— "
            f"來源缺資料，不是停牌。既有檔已保留（只增不減），"
            f"但今天不要拿這份做美股判斷。缺：{', '.join(miss[:15])}"
            f"{' …' if len(miss) > 15 else ''}")

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
        time.sleep(THROTTLE)
    out = pd.DataFrame(rows)
    (RAW / "us").mkdir(parents=True, exist_ok=True)
    out.to_parquet(RAW / "us/fundamentals.parquet", index=False)
    return out


if __name__ == "__main__":
    uni = build_universe(50)
    codes = [c["code"] for c in uni["constituents"]]
    print(f"美股 universe：{len(codes)} 檔 → {', '.join(codes)}")
    # 價量必須涵蓋「曾入選過」的全部代號，不只今天的 50 檔。
    # 少了這一步，PIT 成分股只做了一半：回測知道 2024 年該算 INTC，
    # 但 panel 裡沒有 INTC 的任何一列，那一檔還是靜默消失 ——
    # 生存者偏差原封不動，只是換了個地方藏。台股在 predict.py:72
    # 用 universe.codes_ever() 做了同一件事。
    px_codes = sorted(set(codes) | codes_ever())
    extra = sorted(set(px_codes) - set(codes))
    if extra:
        print(f"　另補 {len(extra)} 檔曾入選但已掉出的：{', '.join(extra)}")
    px = fetch_prices(px_codes)
    print(f"價量：{len(px):,} 列 × {px['code'].nunique()} 檔，"
          f"{px['date'].min().date()} ~ {px['date'].max().date()}")
    fd = fetch_fundamentals(codes)
    print(f"基本面：{fd['PER'].notna().sum()}/{len(fd)} 檔有 PER、"
          f"{fd['rev_yoy'].notna().sum()} 檔有營收成長")
