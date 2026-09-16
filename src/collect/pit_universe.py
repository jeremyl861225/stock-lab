# -*- coding: utf-8 -*-
"""Point-in-time universe —— 用元大 0050 的每日申購買回清單（PCF）重建歷史成分股。

為什麼這是正解，而不是自己用市值排序重建：
  PCF 記錄的是「那一天這檔 ETF 實際持有哪 50 檔、各多少股」。
  它是實際交易的籃子，不是事後推算 —— **零生存者偏差是結構上成立的**。
  相較之下自行重算市值有兩個難處：歷史股數要另外取得，
  而且流通量調整、市值上限等規則無法完全複製。

為什麼不用 TWSE 自己的歷史端點：
  它的 WAF 是累積式的，實測抓 60–80 次就整個 IP 封鎖，且 30 分鐘內不解除。
  PCF API 免認證、無觀測到的限流。

已知限制（建立在資料上時要記得）：
  1. PCF 是 ETF 的申購籃子，在調整日邊界可能與指數本身有差異，且帶現金替代旗標。
  2. 0050 於 2025-06-11 分割，NAV 與單位數序列需調整（個股持股數不受影響）。
  3. 建立單位數本身變過（2003 年 100 萬單位 → 後來 50 萬），
     所以跨時期要用權重而非原始 qty。

注意：此 API 的 TLS 交握與 Python requests 不相容（實測 Max retries），必須走 curl。
"""
from __future__ import annotations
import datetime as dt, json, subprocess, sys, time, uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW, CONFIG

BASE = "https://etfapi.yuantaetfs.com/ectranslation/api/bridge"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/140.0 Safari/537.36")


def fetch_pcf(date: str, ticker: str = "0050") -> dict | None:
    """date: YYYYMMDD。回傳 None 表示該日無資料（非交易日或早於成立日）。"""
    cache = RAW / "pcf" / ticker
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{date}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))

    url = (f"{BASE}?APIType=ETFAPI&CompanyName=YUANTAFUNDS"
           f"&PageName=%2FtradeInfo%2Fpcf%2F{ticker}&DeviceId={uuid.uuid4()}"
           f"&FuncId=PCF%2FDaily&AppName=ETF&Device=3&Platform=ETF"
           f"&ticker={ticker}&date={date}")
    r = subprocess.run(["curl", "-sS", "--max-time", "40", "-H", f"User-Agent: {UA}",
                        "-H", f"Referer: https://www.yuantaetfs.com/tradeInfo/pcf/{ticker}",
                        url], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        j = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    if not j.get("PCF"):
        return None
    f.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return j


def constituents(j: dict) -> list[dict]:
    comp = (j.get("InKind") or {}).get("FundComposition") or []
    return [{"code": c["stkcd"].strip(), "name": c["name"].strip(),
             "qty": float(c.get("qty") or 0)} for c in comp
            if c.get("stkcd", "").strip().isdigit()]


def month_dates(start: str, end: str) -> list[str]:
    s, e = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out, d = [], s.replace(day=15)
    while d <= e:
        out.append(d.strftime("%Y%m%d"))
        d = (d.replace(day=28) + dt.timedelta(days=7)).replace(day=15)
    return out


def build(start: str, end: str) -> dict:
    dates = month_dates(start, end)
    print(f"每月取樣 {len(dates)} 個時點：{dates[0]} ~ {dates[-1]}")
    snaps, ever = {}, set()
    out = CONFIG / "universe"
    out.mkdir(exist_ok=True)
    for i, d in enumerate(dates, 1):
        # 15 日若逢假日會無資料，往前找最近的交易日（最多回推 6 天）
        j, used = None, d
        probe = dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
        for back in range(7):
            cand = (probe - dt.timedelta(days=back)).strftime("%Y%m%d")
            j = fetch_pcf(cand)
            if j:
                used = cand
                break
            time.sleep(0.3)
        if not j:
            print(f"  [{i}/{len(dates)}] {d}: 往前 7 天皆無資料")
            continue
        cons = constituents(j)
        trand = (j.get("PCF") or {}).get("trandate", d)
        if not cons:
            print(f"  [{i}/{len(dates)}] {d}: 0 檔")
            continue
        total = sum(c["qty"] for c in cons) or 1.0
        uni = {"as_of": trand, "built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
               "market": "TW", "method": "yuanta_0050_pcf_basket", "size": len(cons),
               "constituents": [{"code": c["code"], "name": c["name"],
                                 "industry": "", "mcap": c["qty"],
                                 "weight": c["qty"] / total} for c in cons]}
        (out / f"{trand}.json").write_text(
            json.dumps(uni, ensure_ascii=False, indent=2), encoding="utf-8")
        snaps[trand] = [c["code"] for c in cons]
        ever |= set(snaps[trand])
        print(f"  [{i}/{len(dates)}] {d} → {trand}: {len(cons)} 檔")
        time.sleep(0.5)

    meta = {"built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "source": "yuanta 0050 PCF daily basket",
            "snapshots": len(snaps), "codes_ever": sorted(ever),
            "dates": sorted(snaps)}
    (CONFIG / "pit_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"\n快照 {len(snaps)} 份｜曾入選過 {len(ever)} 檔")
    return meta


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else "2024-08-01"
    e = sys.argv[2] if len(sys.argv) > 2 else dt.date.today().isoformat()
    m = build(s, e)
    cur = set(json.loads((CONFIG / "universe_latest.json").read_text(encoding="utf-8")).get(
        "constituents", []) and [c["code"] for c in json.loads(
        (CONFIG / "universe_latest.json").read_text(encoding="utf-8"))["constituents"]])
    ever = set(m["codes_ever"])
    print(f"\n目前 universe {len(cur)} 檔｜PIT 聯集 {len(ever)} 檔")
    print(f"曾入選但現已不在：{sorted(ever - cur)}")
    print(f"現在有但歷史未入選：{sorted(cur - ever)}")
