# -*- coding: utf-8 -*-
"""美股季度財報收集（yfinance）。

為什麼要單獨寫一支，而不是沿用 collect/us.py 的 `.info`：
  `.info` 給的是**當下快照**（PER、revenueGrowth 各一個數字），
  沒有期別、沒有歷史、沒有可用日。一年期判斷要的是
  「毛利率比四季前高了幾個百分點」「資本支出強度有沒有縮手」——
  那些都要季度序列才算得出來，而檢查點更要知道「這個數字是哪天才能看到的」。

三個必須自己處理的坑：

1. **yfinance 只給最近 5–7 季。**
   一次抓回來就丟掉舊的，等於永遠只有一年多的歷史，
   eps 的 TTM 年增（需要 8 季）永遠算不出來。
   所以這裡採**累積式合併**：新抓到的期別併進既有檔案，舊期別一律保留。
   跑久了就長出自己的歷史，不受 yfinance 視窗限制。

2. **金融股沒有毛利率與營業利益。**
   JPM、BRK-B 的損益表裡 Gross Profit／Operating Income 兩列根本不存在，
   跟台股金控是同一個結構性問題。這裡照實記成 null，
   由判斷端決定排除（見 build_1y_us 的說明），不在這裡假裝有值。

3. **外國發行人的幣別不是美元。**
   TSM 的財報在 yfinance 是以新台幣計價（單季資本支出 −5,084 億）。
   本模組只輸出比率（毛利率、資本支出強度、ROE），分子分母同幣別，
   所以不換算也不失真 —— 但**絕對金額一律不要拿來跨公司比較**。
"""
from __future__ import annotations
import datetime as dt, json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW

OUT = RAW / "us_fund"

# yfinance 的列名。同一個概念在不同公司會用不同列名，依序取第一個有值的。
IS_ROWS = {
    "revenue":   ["Total Revenue", "Operating Revenue"],
    "gross":     ["Gross Profit"],
    "op_income": ["Operating Income", "Total Operating Income As Reported"],
    "net_income":["Net Income", "Net Income Common Stockholders",
                  "Net Income From Continuing Operation Net Minority Interest"],
    "eps":       ["Diluted EPS", "Basic EPS"],
}
CF_ROWS = {
    "capex": ["Capital Expenditure", "Purchase Of PPE"],
    "cfo":   ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
}
BS_ROWS = {
    "equity": ["Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"],
    "assets": ["Total Assets"],
}


def _pick(df: pd.DataFrame | None, names: list[str], col) -> float | None:
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            v = df.loc[n, col]
            if isinstance(v, pd.Series):          # 重複列名，取第一個非空
                v = v.dropna()
                v = v.iloc[0] if len(v) else None
            if v is not None and pd.notna(v):
                return float(v)
    return None


def _statements(t, freq: str) -> dict[str, pd.DataFrame | None]:
    if freq == "Q":
        return {"is": t.quarterly_income_stmt, "cf": t.quarterly_cashflow,
                "bs": t.quarterly_balance_sheet}
    return {"is": t.income_stmt, "cf": t.cashflow, "bs": t.balance_sheet}


def _harvest(t, freq: str) -> dict[str, dict]:
    """把一家公司的報表轉成 {期別: 各欄位} —— 期別用 ISO 日期字串當鍵。"""
    s = _statements(t, freq)
    per: dict[str, dict] = {}
    for kind, rows in (("is", IS_ROWS), ("cf", CF_ROWS), ("bs", BS_ROWS)):
        df = s[kind]
        if df is None or getattr(df, "empty", True):
            continue
        for col in df.columns:
            k = str(pd.Timestamp(col).date())
            d = per.setdefault(k, {"period_end": k, "freq": freq})
            for field, names in rows.items():
                v = _pick(df, names, col)
                if v is not None:
                    d[field] = v
    return per


def fetch(codes: list[str], sleep: float = 0.4) -> dict:
    """抓季報與年報並**併進**既有快取。回傳各檔累積後的期別數。"""
    import yfinance as yf
    OUT.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.UTC).isoformat()
    stat = {}
    for c in codes:
        f = OUT / f"{c}.json"
        old = {}
        if f.exists():
            try:
                old = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                old = {}
        q = dict(old.get("quarters") or {})
        a = dict(old.get("annual") or {})
        try:
            t = yf.Ticker(c)
            new_q, new_a = _harvest(t, "Q"), _harvest(t, "A")
        except Exception as e:  # noqa: BLE001
            print(f"  {c} 抓取失敗，沿用既有快取：{e}")
            stat[c] = len(q)
            continue
        # 累積式合併：舊期別永遠保留。同一期別若重抓到，
        # 以新值覆蓋（財報會重編），但**不刪任何期別**。
        for k, v in new_q.items():
            q[k] = {**q.get(k, {}), **v, "seen": q.get(k, {}).get("seen", now)}
        for k, v in new_a.items():
            a[k] = {**a.get(k, {}), **v, "seen": a.get(k, {}).get("seen", now)}
        f.write_text(json.dumps({"code": c, "fetched_utc": now,
                                 "quarters": q, "annual": a},
                                ensure_ascii=False), encoding="utf-8")
        stat[c] = len(q)
        time.sleep(sleep)
    return stat


def is_fresh(codes: list[str], hours: int = 20) -> bool:
    """快取是否夠新。判準用「最舊的一檔」——有一檔沒更新就整批重抓，
    否則橫斷面百分位會拿不同日期的資料互比。"""
    if not OUT.exists():
        return False
    cut = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
    for c in codes:
        f = OUT / f"{c}.json"
        if not f.exists():
            return False
        try:
            ts = dt.datetime.fromisoformat(
                json.loads(f.read_text(encoding="utf-8"))["fetched_utc"])
        except Exception:  # noqa: BLE001
            return False
        if ts < cut:
            return False
    return True


if __name__ == "__main__":
    from collect import us as us_mod
    uni = us_mod.load()
    # ETF 沒有財報，抓了只會拿到空表。
    codes = [c["code"] for c in uni["constituents"] if c.get("industry") != "ETF"]
    if is_fresh(codes) and "--force" not in sys.argv:
        print(f"美股財報快取仍新鮮（{len(codes)} 檔），跳過")
        sys.exit(0)
    st = fetch(codes)
    ok = sum(v > 0 for v in st.values())
    print(f"美股季報：{ok}/{len(codes)} 檔有資料，"
          f"累積期別中位數 {int(pd.Series(list(st.values())).median())} 季")
