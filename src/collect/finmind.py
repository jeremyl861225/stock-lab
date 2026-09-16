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
    # 一年期判斷用的基本面三表。免費層可取，回溯到 2016 年（42 季）。
    # 短期（5／20 日）判斷用不到這些 —— 季報一季才換一次值，
    # 在 20 日尺度上是常數，放進去只會增加共線性而不增加資訊。
    "fin": "TaiwanStockFinancialStatements",
    "bs": "TaiwanStockBalanceSheet",
    "cf": "TaiwanStockCashFlowsStatement",
}


# 各資料集的資料頻率。快取新鮮度要按頻率判斷 ——
# 季報的最後一筆永遠是上一季季末，不可能「≥ 昨天」，
# 用日頻的標準去比，快取永遠不命中、每輪重抓全部、額度瞬間燒光。
FREQ = {"fin": "Q", "bs": "Q", "cf": "Q", "rev": "M"}


def _expected_quarterly(end: str) -> str:
    """end 當天應該已公告的最新季末。用法定期限，寧可低估也不高估。"""
    d = dt.date.fromisoformat(end)
    # (公告上限月, 日) → 對應季末
    for pub_m, pub_d, q_off, q_m, q_d in [
            (3, 31, -1, 12, 31), (5, 15, 0, 3, 31),
            (8, 14, 0, 6, 30), (11, 14, 0, 9, 30)][::-1]:
        if (d.month, d.day) >= (pub_m, pub_d):
            return dt.date(d.year + q_off, q_m, q_d).isoformat()
    return dt.date(d.year - 1, 9, 30).isoformat()


def _expected_monthly(end: str) -> str:
    """end 當天應該已公告的最新月份。月營收次月 10 日前公告。"""
    d = dt.date.fromisoformat(end)
    m = d.month - (1 if d.day >= 10 else 2)
    y = d.year
    while m < 1:
        m += 12; y -= 1
    return dt.date(y, m, 1).isoformat()


def _last_expected(end: str) -> str:
    """end 之前最近的一個工作日。用來判斷快取是否已經是最新。

    週末與國定假日沒有新資料，若拿日曆今天去比對，快取永遠不會命中。
    國定假日無法從這裡判斷，所以再往前放寬一天作為緩衝 ——
    寧可偶爾多抓一次，也不要每天固定撞上額度上限。
    """
    d = dt.date.fromisoformat(end)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return (d - dt.timedelta(days=1)).isoformat()


def fetch(kind: str, code: str, start: str, end: str, refresh: bool = False) -> list[dict]:
    """kind: price|inst|margin。start/end: YYYY-MM-DD。落盤快取。"""
    d = RAW / "finmind" / kind
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{code}.json"
    if f.exists() and not refresh:
        c = json.loads(f.read_text(encoding="utf-8"))
        # 用「快取是否已涵蓋最後一筆資料日」判斷，而不是比對日曆今天。
        # 比對今天的話，end 每天 +1 天 → 隔天全部 miss → 每天固定 300 次請求，
        # 正好撞上 FinMind 免費版的每小時上限。
        rows = c.get("payload") or []
        last = max((r.get("date", "") for r in rows), default="")
        freq = FREQ.get(kind, "D")
        want = (_expected_quarterly(end) if freq == "Q" else
                _expected_monthly(end) if freq == "M" else _last_expected(end))
        if c.get("start", "9999") <= start and last and last >= want:
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
                # 免費層是「每小時」上限，睡 20 秒重試 4 次救不了 ——
                # 只會把剩下的請求全部燒成失敗。額度是整點附近才回補，
                # 所以直接拋出，讓呼叫端決定是要等下個整點還是改天再跑。
                # （落盤快取讓重跑只補缺口，不會重抓已完成的部分。）
                raise RuntimeError(
                    f"FinMind 額度用盡（每小時上限），{kind}/{code} 未取得。"
                    "等下一個整點再跑，已完成的部分會由快取略過。")
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
