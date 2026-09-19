"""K 線圖資料。四個時間尺度，各自降取樣。

為什麼不放進面板 parquet：面板是特徵計算用的，只需要 2.1 年；
塞五年會讓它變 2.5 倍大又拖慢每日建置。圖表資料獨立一份。

為什麼不內嵌進 HTML：103 檔 × 四個尺度，內嵌會讓首次載入多背幾百 KB，
而多數人不會展開每一檔。改成獨立 JSON，第一次展開時才抓、抓完存在記憶體。

降取樣的理由：五年有約 1,250 根日 K，在 340px 寬的手機上一根不到 0.3px，
畫出來是一團色塊而非 K 線。各尺度取「看得清且夠用」的根數：
  20 天  日 K  20 根   —— 可點擊，顯示當日 OHLC 與漲跌幅
  3 個月 日 K  約 63 根
  1 年   週 K  約 52 根
  5 年   月 K  約 60 根
週 K／月 K 用標準合成：開＝區間第一筆開、收＝最後一筆收、高＝區間最高、低＝最低。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RAW, DOCS

SPANS = {"d20": ("D", 20), "m3": ("D", 63), "y1": ("W", 52), "y5": ("M", 60)}


def _round(v: float) -> float:
    """依價格量級取小數位。全存兩位會讓 6,000 元的股票多出無意義的位數，
    而 18 元的股票只留整數又會失真。"""
    if v >= 1000:
        return round(v, 0)
    if v >= 100:
        return round(v, 1)
    return round(v, 2)


def _resample(g: pd.DataFrame, rule: str, n: int) -> pd.DataFrame:
    """降取樣成週／月 K。日 K 直接取最後 n 根。"""
    if rule == "D":
        return g.tail(n)
    freq = {"W": "W-FRI", "M": "ME"}[rule]
    r = (g.set_index("date")
          .resample(freq)
          .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
          .dropna()
          .reset_index())
    return r.tail(n)


def _tw_prices() -> pd.DataFrame:
    d = RAW / "finmind" / "price"
    rows = []
    for f in d.glob("*.json"):
        for r in json.loads(f.read_text(encoding="utf-8")).get("payload", []):
            rows.append((r["stock_id"], r["date"], r.get("open"), r.get("max"),
                         r.get("min"), r.get("close")))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["code", "date", "open", "high", "low", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["close"]).sort_values(["code", "date"]).reset_index(drop=True)
    # 必須還原除權息，理由有二：
    # 一、未還原的五年圖會出現一堆假跌 —— 台積電年殖利率約 4%，
    #     五年累積近 20% 的缺口，圖形會系統性低估長期漲幅。
    # 二、面板卡片上的收盤價是還原過的，K 線若不還原，
    #     最後一根的收盤會對不上卡片，讀者無從對帳。
    from features.panel import adjust_for_corporate_actions
    return adjust_for_corporate_actions(df)


def _us_prices() -> pd.DataFrame:
    """五年回補檔 ＋ 每日檔，兩份合併。

    原本是「有 prices_5y.parquet 就只用它」。但那份是一次性回補的產物，
    每日流程只更新 prices.parquet —— 於是美股 K 線的最後一根永遠停在
    回補那天（實測停在 2026-09-16，而面板卡片已經是 09-18 的收盤）。
    圖與卡片對不上，讀者無從對帳，而且不會有任何錯誤訊息。

    合併時以每日檔為準（同一天兩邊都有就取每日檔），因為它才是
    每天被驗證過的那一份。
    """
    need = {"open", "high", "low", "close"}
    parts = []
    for name in ("prices_5y.parquet", "prices.parquet"):
        f = RAW / "us" / name
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        if not need <= set(df.columns):
            continue
        df["date"] = pd.to_datetime(df["date"])
        df["_src"] = 0 if name.startswith("prices_5y") else 1
        parts.append(df.dropna(subset=["close"]))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = (df.sort_values(["code", "date", "_src"])
            .drop_duplicates(["code", "date"], keep="last")
            .drop(columns="_src"))
    return df.sort_values(["code", "date"])


def build() -> Path:
    # 只產面板會顯示的標的。PIT 宇宙含 17 檔已被剔除的歷史成分股，
    # 它們用於特徵的無偏計算，但不會出現在面板上 —— 送出用不到的資料
    # 只是讓使用者多下載 14%。
    try:
        shown = set(pd.read_parquet(DATA / "briefing.parquet")["code"].astype(str))
    except Exception:  # noqa: BLE001
        shown = None

    out = {"series": {}, "as_of": {}}
    for mk, df in (("TW", _tw_prices()), ("US", _us_prices())):
        if df.empty:
            continue
        out["as_of"][mk] = df["date"].max().strftime("%Y-%m-%d")
        for code, g in df.groupby("code"):
            if shown is not None and str(code) not in shown:
                continue
            g = g[["date", "open", "high", "low", "close"]].copy()
            s = {}
            for key, (rule, n) in SPANS.items():
                r = _resample(g, rule, n)
                if len(r) < 3:
                    continue
                blk = {k: [_round(float(x)) for x in r[k]]
                       for k in ("open", "high", "low", "close")}
                # 只有 20 日尺度需要日期與漲跌幅 —— 那是唯一可點擊的尺度，
                # 其餘尺度存了也用不到，徒增檔案大小。
                if key == "d20":
                    blk["t"] = [d.strftime("%m/%d") for d in r["date"]]
                    prev = g["close"].shift(1).reindex(r.index)
                    blk["p"] = [None if pd.isna(a) or not a else round((c / a - 1) * 100, 2)
                                for c, a in zip(r["close"], prev)]
                s[key] = blk
            if s:
                out["series"][str(code)] = s
    p = DOCS / "charts.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")),
                 encoding="utf-8")
    return p


if __name__ == "__main__":
    p = build()
    d = json.loads(p.read_text(encoding="utf-8"))
    kb = p.stat().st_size / 1024
    print(f"K 線資料：{len(d['series'])} 檔 → {p.name}（{kb:,.0f} KB）")
    print(f"  資料日：{d['as_of']}")
    ex = next(iter(d["series"].values()))
    for k, (rule, n) in SPANS.items():
        if k in ex:
            print(f"  {k:<5}{rule} K　{len(ex[k]['close']):>3} 根")
