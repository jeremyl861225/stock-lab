# -*- coding: utf-8 -*-
"""事件日曆：視窗內有沒有財報、月營收、除權息。

為什麼要有這個（2026-09-19 補上的缺口）：
  這套系統原本完全沒有事件日曆。對 5／20 日期間，「財報是否落在視窗內」
  是最大的遺漏條件變數 —— 它同時決定實現波動與漂移：
  同一檔股票，視窗內有法說會與沒有法說會，是兩個不同的賭局，
  而系統對兩者給出一樣的幅度。

  一個具體例子：2026-09-18 台積電判斷寫「10 月中法說會落在 20 日視窗內」，
  那是人去記的；5 日版的 build 還要靠一行字串替換把那句話改掉
  （judgments/build_20260918.py 的 `if horizon == 5 and ... in w`）。
  人記得住 103 檔裡的一檔，記不住 103 檔。

**這些欄位刻意不進 FEATURE_COLS。**
  未來的財報日是「今天查得到」的資訊，不是「as_of 當天查得到」的 ——
  yfinance 給的是此刻的排程表，沒有歷史版本。放進統計模型的訓練集
  等於讓模型知道未來某天有財報，那是前視偏差。
  它只進 briefing 給人看：人做的是今天這一筆判斷，不回測。

各市場的事件來源：
  台股　月營收（法定次月 10 日前）· 季報法定公告期限 · 除權息（FinMind，若有未來日期）
  美股　財報日（yfinance get_earnings_dates，含未來排程）
  台股的法說會日期沒有免費的結構化來源，用季報法定期限當代理 ——
  它是上限日，實際法說多半更早，所以是保守側（會低估事件密度，不會高估）。

  例外是有美國存託憑證的台廠：2330 與 TSM 是**同一家公司、同一場法說會**，
  而 TSM 那邊 yfinance 給得到確切日期。ADR 對照見下方 `ADR`。
  2026-09-18 的實例：TSM 財報日 2026-10-15，落在 20 日視窗內 —— 這正是
  當天 2330 的判斷裡人工寫下的「10 月中法說會落在 20 日視窗內」。
  系統現在自己算得出來，不必靠人記得住 103 檔裡的哪一檔。
"""
from __future__ import annotations
import datetime as dt, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW, DATA
from features.fundamentals import PUB, _pub_date

SRC = RAW / "us_events"
# 台股代號 → 美國存託憑證代號。同一家公司同一場法說會，日期可以共用。
ADR = {"2330": "TSM"}
CACHE_HOURS = 20          # 與 collect/us_fundamentals.py 同一個節奏
EVENT_COLS = ["evt_type", "evt_date", "evt_days", "evt_in_5", "evt_in_20"]


# ── 美股：財報排程 ──────────────────────────────────────────────────────
def fetch_us(codes: list[str], sleep: float = 0.3) -> int:
    """抓未來財報日並落盤。已在快取期限內的不重抓。"""
    import time
    import yfinance as yf
    SRC.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.UTC)
    n = 0
    for c in codes:
        f = SRC / f"{c.replace('/', '_')}.json"
        if f.exists():
            try:
                old = json.loads(f.read_text(encoding="utf-8"))
                age = now - dt.datetime.fromisoformat(old["fetched_at"])
                if age.total_seconds() < CACHE_HOURS * 3600:
                    continue
            except Exception:  # noqa: BLE001
                pass
        dates = []
        try:
            ed = yf.Ticker(c).get_earnings_dates(limit=16)
            if ed is not None and len(ed):
                for d in pd.to_datetime(ed.index):
                    dates.append(pd.Timestamp(d).tz_localize(None).strftime("%Y-%m-%d"))
        except Exception as e:  # noqa: BLE001
            print(f"  {c}: {type(e).__name__} {str(e)[:60]}")
        if not dates:
            continue
        f.write_text(json.dumps({"code": c, "fetched_at": now.isoformat(),
                                 "earnings": sorted(set(dates))},
                                ensure_ascii=False), encoding="utf-8")
        n += 1
        time.sleep(sleep)
    return n


def _us_events(code: str) -> list[tuple[pd.Timestamp, str]]:
    f = SRC / f"{code.replace('/', '_')}.json"
    if not f.exists():
        return []
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    return [(pd.Timestamp(x), "財報") for x in d.get("earnings", [])]


# ── 台股：月營收、季報期限、除權息 ──────────────────────────────────────
def _tw_statutory(as_of: pd.Timestamp, ahead_days: int = 120) -> list[tuple[pd.Timestamp, str]]:
    """未來 ahead_days 內的月營收公告日與季報法定期限（全市場共通）。"""
    out = []
    d = as_of.normalize()
    end = d + pd.Timedelta(days=ahead_days)
    # 月營收：每月 10 日
    m = pd.Timestamp(year=d.year, month=d.month, day=10)
    while m <= end:
        if m > d:
            out.append((m, "月營收"))
        m = (m + pd.offsets.MonthBegin(1)) + pd.Timedelta(days=9)
    # 季報法定期限
    for (mm, dd) in PUB:
        for yr in (d.year, d.year + 1):
            q = pd.Timestamp(year=yr, month=mm, day=dd)
            pub = _pub_date(q)
            if d < pub <= end:
                out.append((pub, "季報期限"))
    return out


def _tw_exdiv(code: str, as_of: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
    d = RAW / "finmind" / "div"
    if not d.exists():
        return []
    out = []
    for f in d.glob("*.json"):
        try:
            payload = json.loads(f.read_text(encoding="utf-8")).get("payload", [])
        except Exception:  # noqa: BLE001
            continue
        for r in payload:
            if str(r.get("stock_id")) != str(code):
                continue
            try:
                t = pd.Timestamp(r["date"])
            except Exception:  # noqa: BLE001
                continue
            if t > as_of:
                out.append((t, "除權息"))
    return out


# ── 組裝 ────────────────────────────────────────────────────────────────
def _biz_days(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """a→b 之間的營業日數（近似交易日；不處理各國假日，誤差最多一兩天）。"""
    return int(np.busday_count(a.date(), b.date()))


def upcoming(codes: pd.Series | list[str], markets: pd.Series | list[str],
             as_of) -> pd.DataFrame:
    """每一檔的下一個事件。回傳 [code] + EVENT_COLS。"""
    as_of = pd.Timestamp(as_of)
    codes = list(codes)
    markets = list(markets)
    tw_common = _tw_statutory(as_of)
    rows = []
    for c, mk in zip(codes, markets):
        if mk == "US":
            ev = _us_events(c)
        else:
            ev = tw_common + _tw_exdiv(c, as_of)
            # 有 ADR 的台廠：法說會日期用 ADR 那邊的確切日期，
            # 比「季報法定期限」這個上限日精確得多。
            ev += [(t, "財報") for t, _ in _us_events(ADR[c])] if c in ADR else []
        ev = sorted([(t, k) for t, k in ev if t > as_of])
        if not ev:
            rows.append({"code": c, "evt_type": None, "evt_date": None,
                         "evt_days": np.nan, "evt_in_5": False, "evt_in_20": False})
            continue
        t, kind = ev[0]
        n = _biz_days(as_of, t)
        # 視窗內是否還有別的事件（例如 20 日內同時有月營收與季報期限）
        kinds = sorted({k for tt, k in ev if _biz_days(as_of, tt) <= 20})
        rows.append({"code": c, "evt_type": kind, "evt_date": t.strftime("%Y-%m-%d"),
                     "evt_days": n,
                     "evt_in_5": n <= 5,
                     "evt_in_20": n <= 20,
                     "evt_types_20": "、".join(kinds) if kinds else None})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import glob
    pnl = pd.read_parquet(DATA / "features/panel.parquet")
    as_of = pnl["date"].max()
    us_codes = sorted(pnl[pnl["market"] == "US"]["code"].unique()) if "market" in pnl else []
    if "--fetch" in sys.argv and us_codes:
        print(f"抓美股財報排程（{len(us_codes)} 檔，快取 {CACHE_HOURS} 小時）…")
        print(f"  新抓 {fetch_us(us_codes)} 檔")
    last = pnl.sort_values("date").groupby("code").last().reset_index()
    e = upcoming(last["code"], last.get("market", pd.Series(["TW"] * len(last))), as_of)
    e = e.merge(last[["code"] + (["market"] if "market" in last else [])], on="code")
    print(f"\n事件日曆 as_of {as_of.date()}：{len(e)} 檔")
    for mk, g in e.groupby("market") if "market" in e else [("ALL", e)]:
        print(f"\n{mk}: 5 日內有事件 {int(g['evt_in_5'].sum())} 檔、"
              f"20 日內 {int(g['evt_in_20'].sum())} 檔")
        print(g.dropna(subset=["evt_date"]).sort_values("evt_days")
              [["code", "evt_type", "evt_date", "evt_days"]].head(6).to_string(index=False))
