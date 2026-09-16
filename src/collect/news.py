"""新聞收集。

兩個容易毀掉整個 LLM 模型的坑，這裡都處理了：
1. 查詢太寬會抓到八卦（實測「台積電」抓回來的前三則是員工緋聞）。
   → 強制加上財經關鍵字過濾。
2. 新聞也有 point-in-time 問題：預測 as_of 當天時，只能看 as_of 當天
   收盤前已發布的新聞。pubDate 一律保留，讓結算時能稽核。
"""
from __future__ import annotations
import datetime as dt, json, re, sys, time, urllib.parse
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) stock-lab/1.0"}
FIN_TERMS = "股價 OR 營收 OR 法說 OR 外資 OR 財報 OR 目標價 OR 訂單 OR 產能"

def _parse_rss(xml: str) -> list[dict]:
    items = []
    for block in re.findall(r"<item>(.*?)</item>", xml, re.S):
        def grab(tag):
            m = (re.search(rf"<{tag}><!\[CDATA\[(.*?)\]\]></{tag}>", block, re.S)
                 or re.search(rf"<{tag}>(.*?)</{tag}>", block, re.S))
            return m.group(1).strip() if m else ""
        title = re.sub(r"<[^>]+>", "", grab("title"))
        if not title:
            continue
        items.append({"title": title, "link": grab("link"),
                      "pub": grab("pubDate"), "source": grab("source")})
    return items

def _pub_ts(s: str):
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z"):
        try:
            return dt.datetime.strptime(s, fmt).astimezone(dt.UTC)
        except ValueError:
            continue
    return None

def stock_news(code: str, name: str, as_of: str, days: int = 7) -> list[dict]:
    """個股近期財經新聞，已濾除 as_of 之後發布的內容。"""
    cp = RAW / "news" / as_of
    cp.mkdir(parents=True, exist_ok=True)
    f = cp / f"{code}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["payload"]

    q = urllib.parse.quote(f"{name} ({FIN_TERMS}) when:{days}d")
    url = f"https://news.google.com/rss/search?q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    items = []
    try:
        r = requests.get(url, headers=UA, timeout=25)
        if r.status_code == 200:
            items = _parse_rss(r.text)
    except Exception as e:  # noqa: BLE001
        print(f"  news fail {code}: {e}")

    cutoff = dt.datetime.strptime(as_of, "%Y%m%d").replace(
        hour=13, minute=30, tzinfo=dt.timezone(dt.timedelta(hours=8))).astimezone(dt.UTC)
    keep = []
    for it in items:
        ts = _pub_ts(it["pub"])
        if ts is None or ts <= cutoff:          # 只留收盤前已發布的
            it["pub_utc"] = ts.isoformat() if ts else None
            keep.append(it)
    keep = keep[:12]
    f.write_text(json.dumps({"fetched_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                             "code": code, "as_of": as_of, "payload": keep},
                            ensure_ascii=False), encoding="utf-8")
    time.sleep(0.4)
    return keep

def market_news(as_of: str) -> list[dict]:
    """大盤層級新聞（宏觀背景）。"""
    cp = RAW / "news" / as_of
    cp.mkdir(parents=True, exist_ok=True)
    f = cp / "_market.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["payload"]
    out = []
    for url in ("https://news.cnyes.com/rss/v1/news/category/tw_stock",
                "https://tw.stock.yahoo.com/rss?category=tw-market"):
        try:
            r = requests.get(url, headers=UA, timeout=25)
            if r.status_code == 200:
                out += _parse_rss(r.text)[:15]
        except Exception:  # noqa: BLE001
            pass
    f.write_text(json.dumps({"fetched_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                             "as_of": as_of, "payload": out}, ensure_ascii=False),
                 encoding="utf-8")
    return out
