"""TWSE 官方資料收集層。

原則：
1. 每一筆抓下來的原始資料都落盤（data/raw/），帶抓取時間戳 —— 事後可重現。
2. 已抓過的日期不重抓（快取），避免對官方站台施壓。
3. 只抓「已公布」的資料，絕不推估未來。
"""
from __future__ import annotations

import json
import random
import time
import datetime as dt
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW

OPENAPI = "https://openapi.twse.com.tw/v1"
RWD = "https://www.twse.com.tw/rwd/zh"
UA = {"User-Agent": "stock-lab/1.0 (research; contact via github)"}
SLEEP = 0.6  # 對官方站台的禮貌間隔


def _get(url: str, params: dict | None = None, retries: int = 5):
    """TWSE 限流時會回 307 並重導到「因為安全性考量」頁，而非 429。
    必須關掉 allow_redirects 才看得到，且要指數退避，否則會整批失敗。"""
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=30,
                             allow_redirects=False)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (302, 307, 429, 503):
                last = f"rate-limited HTTP {r.status_code}"
                time.sleep(4 * (i + 1) + random.random() * 2)
                continue
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = repr(e)
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"GET failed {url} {params}: {last}")


def _cache_path(kind: str, date: str) -> Path:
    p = RAW / kind
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{date}.json"


def _save(kind: str, date: str, payload):
    p = _cache_path(kind, date)
    p.write_text(json.dumps({
        "fetched_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "kind": kind, "date": date, "payload": payload,
    }, ensure_ascii=False), encoding="utf-8")
    return p


def daily_quotes(date: str) -> list[dict]:
    """某交易日全市場行情。date: YYYYMMDD。回傳 [] 代表非交易日。"""
    cp = _cache_path("mi_index", date)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["payload"]

    d = _get(f"{RWD}/afterTrading/MI_INDEX",
             {"date": date, "type": "ALL", "response": "json"})
    rows = []
    if d.get("stat") == "OK":
        # 全市場個股表：欄位數最多、且首欄為 4 碼代號的那張表
        for tbl in d.get("tables", []):
            fields = tbl.get("fields") or []
            if "證券代號" in fields and "收盤價" in fields:
                idx = {f: i for i, f in enumerate(fields)}
                for r in tbl.get("data", []):
                    rows.append({
                        "code": r[idx["證券代號"]].strip(),
                        "name": r[idx["證券名稱"]].strip(),
                        "volume": r[idx.get("成交股數", 2)],
                        "value": r[idx.get("成交金額", 4)],
                        "open": r[idx.get("開盤價", 5)],
                        "high": r[idx.get("最高價", 6)],
                        "low": r[idx.get("最低價", 7)],
                        "close": r[idx["收盤價"]],
                    })
                break
    _save("mi_index", date, rows)
    time.sleep(SLEEP)
    return rows


def institutional(date: str) -> list[dict]:
    """三大法人買賣超（單位：股）。"""
    cp = _cache_path("t86", date)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["payload"]

    d = _get(f"{RWD}/fund/T86",
             {"date": date, "selectType": "ALL", "response": "json"})
    rows = []
    if d.get("stat") == "OK":
        fields = d.get("fields", [])
        idx = {f: i for i, f in enumerate(fields)}

        def pick(r, *names):
            for n in names:
                if n in idx:
                    return r[idx[n]]
            return "0"

        for r in d.get("data", []):
            rows.append({
                "code": r[idx["證券代號"]].strip(),
                "foreign": pick(r, "外陸資買賣超股數(不含外資自營商)", "外資買賣超股數"),
                "trust": pick(r, "投信買賣超股數"),
                "dealer": pick(r, "自營商買賣超股數"),
                "total": pick(r, "三大法人買賣超股數"),
            })
    _save("t86", date, rows)
    time.sleep(SLEEP)
    return rows


def valuations(date: str) -> list[dict]:
    """本益比／殖利率／股價淨值比（僅當日快照，官方不提供歷史回補）。"""
    cp = _cache_path("bwibbu", date)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["payload"]
    d = _get(f"{OPENAPI}/exchangeReport/BWIBBU_ALL")
    rows = [{"code": r.get("Code", "").strip(), "pe": r.get("PEratio", ""),
             "yield": r.get("DividendYield", ""), "pb": r.get("PBratio", ""),
             "date_roc": r.get("Date", "")} for r in d]
    _save("bwibbu", date, rows)
    time.sleep(SLEEP)
    return rows


def listed_companies() -> list[dict]:
    """上市公司基本資料（含已發行股數），用於計算市值。"""
    today = dt.datetime.now().strftime("%Y%m%d")
    cp = _cache_path("basic", today)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["payload"]
    d = _get(f"{OPENAPI}/opendata/t187ap03_L")
    key_shares = "已發行普通股數或TDR原股發行股數"
    rows = [{
        "code": r.get("公司代號", "").strip(),
        "name": r.get("公司簡稱", "").strip(),
        "industry": r.get("產業別", "").strip(),
        "shares": r.get(key_shares, "") or "",
        "capital": r.get("實收資本額", "") or "",
        "par": r.get("普通股每股面額", "") or "",
    } for r in d]
    _save("basic", today, rows)
    return rows


def stock_month(code: str, yyyymm: str) -> list[dict]:
    """個股單月日成交資料。yyyymm: YYYYMM。比 MI_INDEX 小兩個數量級。"""
    cp = _cache_path(f"stock_day/{code}", yyyymm)
    if cp.exists():
        return json.loads(cp.read_text(encoding="utf-8"))["payload"]
    d = _get(f"{RWD}/afterTrading/STOCK_DAY",
             {"date": f"{yyyymm}01", "stockNo": code, "response": "json"})
    rows = []
    if d.get("stat") == "OK":
        for r in d.get("data", []):
            roc = r[0].split("/")
            if len(roc) != 3:
                continue
            ad = f"{int(roc[0]) + 1911}{roc[1]}{roc[2]}"
            rows.append({"date": ad, "code": code, "volume": r[1], "value": r[2],
                         "open": r[3], "high": r[4], "low": r[5], "close": r[6]})
    _save(f"stock_day/{code}", yyyymm, rows)
    return rows
