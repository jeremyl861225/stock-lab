# -*- coding: utf-8 -*-
"""美股季報走 SEC EDGAR XBRL companyfacts —— 拿到**真正的申報日**。

為什麼要在 yfinance 之外再寫一支（兩者的分工見檔尾 `__main__` 的輸出）：

1. **可用日從「法定上限」變成「真的哪天公布」。**
   `features/us_fundamentals.py` 原本用 10-Q 季末後 40 天、10-K 後 60 天推估，
   那是法律容許的最晚一天；實務上多數公司提早 2–3 週。這個方向的偏差是
   保守的（比真實世界晚知道），不像「拿期末日當可用日」會讓回測變漂亮 ——
   但它同樣是偏差，而且它讓檢查點的「前提何時翻掉」整整晚兩三週。
   companyfacts 每一筆 fact 自帶 `filed`，那是 EDGAR 收件日，不必推估。

2. **歷史深度從 5–7 季變成 15 年以上。**
   實測 AAPL 的 GrossProfit 有 338 筆、回到 2007-09-29。
   `gm_self_pct`（毛利率相對自身歷史的百分位）要 8 季才生效，
   美股原本得等 `data/raw/us_fund/` 自己一季一季長出來。

**只取原始申報值，不取重編值。**
同一期別會在後續每一份 10-Q／10-K 裡被重報（AAPL 2025-06-28 那季出現兩次）。
一律取 `filed` 最小的那一筆 —— 那才是當時看得到的數字。拿重編後的值去填
當時的位置是典型的前視：重編本身就是後來才發生的事。

**銀行的營收與 yfinance 不是同一個定義。**
JPM 的損益表沒有「Total Revenue」，這裡取 `RevenuesNetOfInterestExpense`
（2026Q2 573.5 億），yfinance 給的是 528.5 億 —— 兩者都對，但算的不是同一件事。
金融股本來就不在一年期判斷內（見 `HANDOFF.md` 九），這裡照實記，
但**兩個來源對金融股不可互換**，混用會讓 rev_yoy 在換源那一季憑空跳一段。

**IFRS 申報人不在覆蓋範圍，而且不假裝有。**
TSM 的 companyfacts 只有 `ifrs-full`，us-gaap 概念一個都沒有（實測 0 個）。
本模組對它回傳空表，由上游退回 yfinance 並逐列標記來源。

**現金流量表是年初至今累計，損益表不是。**
實測 AAPL 的 NetCashProvidedByUsedInOperatingActivities 期間長度為
3／6／9／12 個月，而 GrossProfit 有 208 筆單季值。所以：
  現金流 → 一律「本期累計 − 前期累計」還原單季
  損益表 → 優先取單季 fact，只有財年最後一季（沒有第四份 10-Q）才用年報減前三季

**不靠 fact 的 `fy`／`fp` 欄位判斷期別。** 那兩欄指的是**該份申報**的財年與期別，
不是這筆數字自己的期間；10-K 裡的去年同期比較數會帶著今年的 fy。
一律用 start／end 的日期自己分組。

SEC 的 fair access 規定要求 User-Agent 帶聯絡方式，否則 403。
走環境變數 `SEC_UA`，不寫進 repo。
"""
from __future__ import annotations
import datetime as dt, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW

OUT = RAW / "sec_fund"
TICKER_CACHE = RAW / "sec_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SLEEP = 0.15          # SEC 允許 10 req/s，這裡跑約 6.7 req/s

# 同一個概念在不同年代／不同公司用不同標籤，依序取第一個有值的。
# 實測：AAPL 走 RevenueFromContractWithCustomer...（2018 年新收入準則之後），
# NVDA 走 Revenues；兩者都得有，否則各缺一半歷史。
DURATION = {
    "revenue":    ["RevenueFromContractWithCustomerExcludingAssessedTax",
                   "Revenues", "SalesRevenueNet",
                   "RevenueFromContractWithCustomerIncludingAssessedTax",
                   # 銀行的損益表沒有「營收」，用扣息後淨收入。實測 JPM 三個
                   # 常用標籤一個都不是，只有這個有值。
                   "RevenuesNetOfInterestExpense"],
    "gross":      ["GrossProfit"],
    "op_income":  ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss",
                   "ProfitLoss"],
    "eps":        ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
}
# 有些公司不揭露毛利小計，只揭露成本 —— 實測 GOOGL 的 companyfacts
# 裡 GrossProfit 一筆都沒有，毛利率因此整欄是空的，而它是一年期判斷的主訊號。
# 這時用「營收 − 成本」自己算。順序與 DURATION 同義：先寫的贏。
COST = ["CostOfRevenue", "CostOfGoodsAndServicesSold",
        "CostOfGoodsSold", "CostOfServices"]

# 現金流：累計制，需還原單季
CUMULATIVE = {
    "cfo":   ["NetCashProvidedByUsedInOperatingActivities",
              "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment",
              "PaymentsToAcquireProductiveAssets"],
}
# 時點值：只有 end，沒有 start
INSTANT = {
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
}
SHARES = ["EntityCommonStockSharesOutstanding"]   # dei 分類，PIT 股數

# **只採信定期報表。**
# 實測 CAT 的 FY2021 淨利在 `NetIncomeLoss` 底下只有一筆，來源是 2026-04-30 的
# DEF 14A（委託書的薪酬對照表要揭露前五年淨利）—— 照單全收的話，那一季的
# 「首次公布日」會變成四年後，於是 2021 Q4 在回測裡整整晚四年才可用。
# 委託書、8-K、S-1 都不是財務報表，一律不採。
PERIODIC = {"10-K", "10-Q", "20-F", "40-F",
            "10-K/A", "10-Q/A", "20-F/A", "40-F/A"}

# 期間長度分類（天）。財報季長度在 84~98 天之間浮動（52/53 週制），
# 區間刻意開寬，但不能寬到讓 6 個月被當成 3 個月。
SPANS = {1: (75, 115), 2: (165, 205), 3: (255, 295), 4: (330, 400)}


def _ua() -> str:
    ua = os.getenv("SEC_UA", "").strip()
    if not ua or "@" not in ua:
        raise RuntimeError(
            "SEC 需要 User-Agent 帶聯絡信箱，否則一律 403。\n"
            "  export SEC_UA='stock-lab research your@email.com'\n"
            "（不要寫進 repo；GitHub Actions 設成 secret）")
    return ua


def _get(url: str, retries: int = 3):
    import requests
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, headers={"User-Agent": _ua(),
                                           "Accept-Encoding": "gzip, deflate"},
                             timeout=60)
            if r.status_code == 404:
                return None               # 這家公司沒有 XBRL，不是錯誤
            r.raise_for_status()
            return r.json()
        except Exception as e:            # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"SEC 取得失敗 {url}：{last}")


def ticker_map(refresh: bool = False) -> dict[str, int]:
    """ticker → CIK。SEC 的對照表約 800KB，快取在本機。"""
    if not refresh and TICKER_CACHE.exists():
        try:
            return {k: int(v) for k, v in
                    json.loads(TICKER_CACHE.read_text(encoding="utf-8")).items()}
        except Exception:  # noqa: BLE001
            pass
    d = _get(TICKERS_URL) or {}
    m = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in d.values()}
    TICKER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TICKER_CACHE.write_text(json.dumps(m), encoding="utf-8")
    return m


def _span(start: str, end: str) -> int | None:
    """回傳這筆 fact 涵蓋幾個季度（1~4），對不上任何一格就回 None。"""
    n = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
    for k, (lo, hi) in SPANS.items():
        if lo <= n <= hi:
            return k
    return None


def _first_filed(rows: list[dict]) -> dict:
    """同一期間的多個版本取 `filed` 最小的 —— 當時公布的那一版。"""
    return min(rows, key=lambda r: (r["filed"], r.get("accn", "")))


SPLICE_TOL = 0.01      # 兩個標籤要能接起來，重疊期間的值必須差在 1% 內


def _one_tag(facts: dict, name: str, instant: bool) -> dict:
    """單一標籤 → {鍵: {val, filed}}。鍵：時點值用 end，期間值用 (start, end)。"""
    node = facts.get(name)
    if not node:
        return {}
    buck: dict = {}
    for unit_rows in node.get("units", {}).values():
        for r in unit_rows:
            if "val" not in r or "filed" not in r:
                continue
            if r.get("form") not in PERIODIC:
                continue
            if instant:
                k = r["end"]
            else:
                if "start" not in r:
                    continue
                k = (r["start"], r["end"])
            buck.setdefault(k, []).append(r)
    return {k: {"val": float(_first_filed(v)["val"]), "filed": _first_filed(v)["filed"]}
            for k, v in buck.items()}


def _collect(facts: dict, names: list[str], instant: bool) -> tuple[dict, list[str]]:
    """把候選標籤接成一條序列，**但只在接得起來的時候才接**。

    為什麼不能像 yfinance 那支一樣「先寫的贏、後面補洞」：
      實測 AXP 的 `RevenueFromContractWithCustomerExcludingAssessedTax`
      只從 2017 開始、而且對金融股只算合約收入（單季約 60 億），
      2016 以前則走 `Revenues`（總收入淨額，單季約 80 億）。
      補洞式合併會把這兩段接成同一條序列，於是 2017Q1 的 rev_yoy
      憑空出現 −25% —— 公司什麼事都沒發生，只是換了個 XBRL 標籤。
      四季加總對年報也會差 32.5%（守恆檢查抓到的就是這一筆）。

    規則：
      1. 覆蓋最多期別的標籤當主序列（同票數時依 `names` 的順序）。
      2. 其餘標籤只有在**與主序列重疊的期別上數值一致**（差 <1%）時才併入，
         用來延長歷史。沒有重疊期別就無從驗證，一律不併。
      3. 併不進來的標籤直接丟掉，那些期別留空 —— 寧可少一段歷史，
         不要一段換過定義的歷史。

    回傳 (值, 實際採用的標籤)。標籤要寫進輸出，事後才查得出某一段是誰給的。
    """
    tagged = [(n, _one_tag(facts, n, instant)) for n in names]
    tagged = [(n, d) for n, d in tagged if d]
    if not tagged:
        return {}, []
    tagged.sort(key=lambda t: (-len(t[1]), names.index(t[0])))
    out = dict(tagged[0][1])
    used = [tagged[0][0]]
    for name, d in tagged[1:]:
        both = [k for k in d if k in out]
        if not both:
            continue                      # 沒有重疊 → 驗不了 → 不接
        ok = all(abs(d[k]["val"] - out[k]["val"])
                 <= SPLICE_TOL * max(abs(out[k]["val"]), 1.0) for k in both)
        if not ok:
            continue
        for k, v in d.items():
            out.setdefault(k, v)
        used.append(name)
    return out, used


def _days(k: tuple[str, str]) -> int:
    return (dt.date.fromisoformat(k[1]) - dt.date.fromisoformat(k[0])).days


def _by_end(dur: dict, span: int) -> dict[str, tuple[tuple[str, str], dict]]:
    """把某個期間長度的 fact 依**期末日**收攏，一個期末只留一筆。

    同一個期末常有不只一個起始日。實測 GS 的 2011 財年營運現金流有兩筆：
      (2010-12-31 → 2011-12-31) 365 天　216.45 億　filed 2012-02-28  ← 當年的 10-K
      (2011-01-01 → 2011-12-31) 364 天　225.01 億　filed 2014-02-28  ← 兩年後重編
    取哪一筆要有明確規則，否則就看 dict／set 的走訪順序 —— 而 Python 的
    字串雜湊每個行程都不一樣，於是**同一份輸入每次跑出不同答案**，
    GS 的檔案因此每天都進 commit。這正是本模組「取最早申報」那條規則
    要涵蓋的情形，只是漏了同一期末有多個起始日這一種。

    排序鍵三層，全部確定：最早申報 → 天數最接近標準長度 → 起始日字典序。
    """
    target = {1: 91, 2: 182, 3: 273, 4: 365}[span]
    out: dict[str, tuple[tuple[str, str], dict]] = {}
    for k in sorted(k for k in dur if _span(*k) == span):
        e = k[1]
        cur = (dur[k]["filed"], abs(_days(k) - target), k[0])
        if e not in out:
            out[e] = (k, dur[k])
        else:
            prev_k = out[e][0]
            prev = (out[e][1]["filed"], abs(_days(prev_k) - target), prev_k[0])
            if cur < prev:
                out[e] = (k, dur[k])
    return out


def _fiscal_years(dur: dict) -> list[tuple[str, str]]:
    """財年區間 [(start, end), ...]，一個期末只留一筆，按 end 排序。"""
    return [k for _e, (k, _v) in sorted(_by_end(dur, 4).items())]


def _quarters_from_duration(dur: dict) -> dict[str, dict]:
    """損益表：優先單季 fact；財年最後一季用年報減同財年前三季。"""
    out: dict[str, dict] = {}
    for _e, (k, v) in sorted(_by_end(dur, 1).items()):
        out[k[1]] = {"val": v["val"], "filed": v["filed"], "derived": False}
    for fs, fe in _fiscal_years(dur):
        if fe in out:
            continue
        inner = [(k, v) for k, v in dur.items()
                 if _span(*k) == 1 and fs <= k[0] and k[1] < fe]
        # 同財年內的單季必須剛好三筆，否則寧可不推 —— 少一筆就會多算一季。
        if len(inner) != 3:
            continue
        fy = dur[(fs, fe)]
        val = fy["val"] - sum(v["val"] for _, v in inner)
        filed = max([fy["filed"], *(v["filed"] for _, v in inner)])
        out[fe] = {"val": val, "filed": filed, "derived": True}
    return out


def _quarters_from_cumulative(cum: dict) -> dict[str, dict]:
    """現金流：本期累計 − 前期累計。同一財年的累計 fact 共用同一個起始日。"""
    out: dict[str, dict] = {}
    by_start: dict[str, list] = {}
    for (s, e), v in cum.items():
        if _span(s, e) is not None:
            by_start.setdefault(s, []).append((e, v))
    for s in sorted(by_start):
        rows = sorted(by_start[s])
        prev_val, prev_filed = 0.0, None
        for e, v in rows:
            val = v["val"] - prev_val
            filed = v["filed"] if prev_filed is None else max(v["filed"], prev_filed)
            # 同一期末可能由多個財年的累計序列推出來，取先公布的那個
            if e not in out or filed < out[e]["filed"]:
                out[e] = {"val": val, "filed": filed, "derived": prev_val != 0.0}
            prev_val, prev_filed = v["val"], v["filed"]
    return out


def parse(cf: dict, code: str) -> dict:
    """companyfacts → 與 `data/raw/us_fund/` 同構的季別字典，外加 `filed`。"""
    facts = (cf.get("facts") or {})
    ug = facts.get("us-gaap") or {}
    dei = facts.get("dei") or {}
    quarters: dict[str, dict] = {}
    annual: dict[str, dict] = {}

    def _put(store: dict, period: str, field: str, val: float, filed: str,
             derived: bool) -> None:
        d = store.setdefault(period, {"period_end": period})
        d[field] = val
        d.setdefault("_filed", []).append(filed)
        if derived:
            d.setdefault("derived", []).append(field)

    tags: dict[str, list[str]] = {}
    for field, names in DURATION.items():
        dur, tags[field] = _collect(ug, names, instant=False)
        for e, v in _quarters_from_duration(dur).items():
            _put(quarters, e, field, v["val"], v["filed"], v["derived"])
        for (s, e) in _fiscal_years(dur):
            v = dur[(s, e)]
            _put(annual, e, field, v["val"], v["filed"], False)

    # 毛利補算：只補洞，已經有 GrossProfit 的期別不動。
    cost_dur, tags["cost"] = _collect(ug, COST, instant=False)
    if cost_dur:
        cost_q = _quarters_from_duration(cost_dur)
        rev_q = {e: d.get("revenue") for e, d in quarters.items()}
        for e, cv in cost_q.items():
            if e in quarters and quarters[e].get("gross") is None \
                    and rev_q.get(e) is not None:
                _put(quarters, e, "gross", rev_q[e] - cv["val"], cv["filed"], True)

    for field, names in CUMULATIVE.items():
        cum, tags[field] = _collect(ug, names, instant=False)
        for e, v in _quarters_from_cumulative(cum).items():
            _put(quarters, e, field, v["val"], v["filed"], v["derived"])
        for (s, e) in _fiscal_years(cum):
            v = cum[(s, e)]
            _put(annual, e, field, v["val"], v["filed"], False)

    for field, names in INSTANT.items():
        inst, tags[field] = _collect(ug, names, instant=True)
        for e, v in inst.items():
            if e in quarters:
                _put(quarters, e, field, v["val"], v["filed"], False)
            if e in annual:
                _put(annual, e, field, v["val"], v["filed"], False)

    # 一列有兩個日期，因為它們回答不同的問題：
    #   filed_first = 各欄位首次公布日的**最小值** —— 這一季的財報是哪天送件的
    #   filed       = **最大值** —— 這一列用到的每一個數字都公開了的那天
    # 正常情況兩者相同（同一份 10-Q 出來的）。會差很多的是**標籤改版**：
    # 2018 年收入準則換標籤那兩年，用 max 算出來的落後天數中位數是 396 天，
    # 而那些數字當年就在財報上，只是換了個 XBRL 標籤才重新出現。
    # 兩個都存，由 features/us_fundamentals.py 決定採信哪一個並逐列標記，
    # 這裡不替它決定 —— 收集器只負責事實。
    for store, freq in ((quarters, "Q"), (annual, "A")):
        for d in store.values():
            f = d.pop("_filed")
            d["filed"], d["filed_first"] = max(f), min(f)
            d["freq"] = freq

    # PIT 股數：封面頁申報股數，每筆帶 filed → 可還原當時市值
    shares = []
    for e, v in sorted(_collect(dei, SHARES, instant=True)[0].items()):
        shares.append({"end": e, "shares": v["val"], "filed": v["filed"]})

    return {
        "code": code,
        "cik": cf.get("cik"),
        "entity": cf.get("entityName", ""),
        "taxonomy": "us-gaap" if ug else (next(iter(facts), "") or ""),
        "tags": {k: v for k, v in tags.items() if v},
        "quarters": quarters,
        "annual": annual,
        "shares": shares,
    }


def fetch(codes: list[str], sleep: float = SLEEP) -> dict[str, int]:
    """抓取並落地。回傳各檔的季別數；IFRS／查無 CIK 者為 0。"""
    OUT.mkdir(parents=True, exist_ok=True)
    tm = ticker_map()
    now = dt.datetime.now(dt.UTC).isoformat()
    stat: dict[str, int] = {}
    for c in codes:
        cik = tm.get(c.upper()) or tm.get(c.upper().replace("-", "."))
        if not cik:
            print(f"  {c} 查無 CIK，跳過")
            stat[c] = 0
            continue
        cf = _get(FACTS_URL.format(cik=cik))
        time.sleep(sleep)
        if not cf:
            print(f"  {c} 無 companyfacts，跳過")
            stat[c] = 0
            continue
        d = parse(cf, c)
        if not d["quarters"]:
            print(f"  {c} 無 us-gaap 季別（taxonomy={d['taxonomy']}），跳過")
            stat[c] = 0
            continue
        f = OUT / f"{c}.json"
        # 內容沒變就不重寫。季報一季才動一次，而 `fetched_utc` 天天都不一樣 ——
        # 照寫的話這 49 個檔每天都進 commit，把真正的財報更新淹沒在雜訊裡，
        # 而且 roll_1y 的基礎雜湊會跟著檔案 mtime 一起漂。
        prev = None
        if f.exists():
            try:
                prev = json.loads(f.read_text(encoding="utf-8"))
                prev.pop("fetched_utc", None)
            except Exception:  # noqa: BLE001
                prev = None
        if prev != d:
            d["fetched_utc"] = now
            f.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        stat[c] = len(d["quarters"])
    return stat


def is_fresh(codes: list[str], hours: int = 24) -> bool:
    """快取是否夠新。SEC 的資料一季才動一次，不必每天重抓 49 檔。"""
    if not OUT.exists():
        return False
    cut = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
    newest = None
    for c in codes:
        f = OUT / f"{c}.json"
        if not f.exists():
            continue                      # IFRS 申報人本來就不會有檔
        ts = dt.datetime.fromtimestamp(f.stat().st_mtime, dt.UTC)
        newest = ts if newest is None else max(newest, ts)
    return newest is not None and newest >= cut


if __name__ == "__main__":
    from collect import us as us_mod
    uni = us_mod.load()
    codes = [c["code"] for c in uni["constituents"] if c.get("industry") != "ETF"]
    if len(sys.argv) > 1 and sys.argv[1] != "--force":
        codes = sys.argv[1:]
    if is_fresh(codes) and "--force" not in sys.argv:
        print(f"SEC 季報快取仍新鮮（{len(codes)} 檔），跳過")
        sys.exit(0)
    st = fetch(codes)
    ok = {k: v for k, v in st.items() if v}
    print(f"\nSEC 季報：{len(ok)}/{len(codes)} 檔有資料")
    if ok:
        import statistics
        print(f"  累積期別中位數 {int(statistics.median(ok.values()))} 季"
              f"（yfinance 一次只給 5–7 季）")
    miss = [k for k, v in st.items() if not v]
    if miss:
        print(f"  無資料：{' '.join(miss)}　← 這些仍走 yfinance")
