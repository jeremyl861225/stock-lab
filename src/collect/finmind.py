"""FinMind 資料源（價量／法人／融資券）。

為什麼不用 TWSE 官方逐日端點抓歷史：
  TWSE 的 STOCK_DAY 一次只給「一檔一個月」，兩年 50 檔要 1,250 個請求，
  而且站台有嚴格限流 —— 實測會先回 307（重導到「因為安全性考量」頁），
  密集一點就變 428，接著整個 IP 被冷凍。
  FinMind 一次請求就能取回一檔的完整兩年歷史，150 個請求解決全部。
  TWSE openapi.twse.com.tw（不同主機、不受影響）仍用於 universe 的市值計算。
"""
from __future__ import annotations
import datetime as dt, json, os, sys, time
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW

API = "https://api.finmindtrade.com/api/v4/data"
TOKEN = os.getenv("FINMIND_TOKEN", "")

DATASETS = {
    "price": "TaiwanStockPrice",
    "inst": "TaiwanStockInstitutionalInvestorsBuySell",
    "margin": "TaiwanStockMarginPurchaseShortSale",
    "div": "TaiwanStockDividendResult",
    "per": "TaiwanStockPER",
    "rev": "TaiwanStockMonthRevenue",
}


def fetch(kind: str, code: str, start: str, end: str, refresh: bool = False) -> list[dict]:
    """kind: price|inst|margin。start/end: YYYY-MM-DD。落盤快取。"""
    d = RAW / "finmind" / kind
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{code}.json"
    if f.exists() and not refresh:
        c = json.loads(f.read_text(encoding="utf-8"))
        if c.get("start") <= start and c.get("end") >= end:
            return c["payload"]

    params = {"dataset": DATASETS[kind], "data_id": code,
              "start_date": start, "end_date": end}
    if TOKEN:
        params["token"] = TOKEN
    last = None
    for i in range(4):
        try:
            r = requests.get(API, params=params, timeout=90)
            if r.status_code == 200:
                j = r.json()
                if j.get("msg") in ("success", None):
                    rows = j.get("data", [])
                    f.write_text(json.dumps({
                        "fetched_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "kind": kind, "code": code, "start": start, "end": end,
                        "payload": rows}, ensure_ascii=False), encoding="utf-8")
                    return rows
                last = j.get("msg")
            elif r.status_code == 402:
                last = "FinMind 額度用盡（每小時上限）"
                time.sleep(20)
                continue
            else:
                last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(f"FinMind {kind} {code}: {last}")


def backfill(codes: list[str], start: str, end: str, refresh: bool = False) -> dict:
    stat = {"ok": 0, "fail": 0, "errors": []}
    t0 = time.time()
    for n, c in enumerate(codes, 1):
        for kind in ("price", "inst", "margin", "div", "per", "rev"):
            try:
                fetch(kind, c, start, end, refresh)
                stat["ok"] += 1
            except Exception as e:  # noqa: BLE001
                stat["fail"] += 1
                stat["errors"].append(f"{c}/{kind}: {e}")
            time.sleep(0.15)
        if n % 10 == 0:
            print(f"  {n}/{len(codes)} 檔完成 ({time.time()-t0:.0f}s)", flush=True)
    print(f"完成：成功 {stat['ok']}、失敗 {stat['fail']}、{time.time()-t0:.0f}s", flush=True)
    return stat


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from collect import universe
    uni = universe.load()
    codes = [c["code"] for c in uni["constituents"]]
    end = dt.date.today().strftime("%Y-%m-%d")
    start = (dt.date.today() - dt.timedelta(days=760)).strftime("%Y-%m-%d")
    print(f"FinMind 回補 {len(codes)} 檔 × 3 資料集，{start} ~ {end}")
    s = backfill(codes, start, end)
    if s["errors"][:5]:
        print("前幾個錯誤:", *s["errors"][:5], sep="\n  ")
