# -*- coding: utf-8 -*-
"""美股 point-in-time 成分股 —— 用 SEC 的申報股數 × 當時未還原收盤價還原當時的市值。

台股那支（`pit_universe.py`）抄的是元大 0050 每天實際持有什麼，
**零生存者偏差是結構上成立的**。美股沒有對應的籃子可抄，因為本系統的
美股 universe 不是任何指數，而是自訂的「候選池裡市值前 50 大 ＋ 三檔 ETF」
（`collect/us.py`）。所以這裡走的是**同一個定義、換成 point-in-time 的輸入**：

    當時市值 = 當時的申報股數 × 當時的未還原收盤價

兩個輸入都是那一天就看得到的：
  · 股數走 SEC companyfacts 的 `dei:EntityCommonStockSharesOutstanding`
    （10-K／10-Q 封面頁的流通股數），每筆帶 `filed`，取 `filed <= 當日` 的最新一筆。
  · 價格走 yfinance 的 **`auto_adjust=False`**。市值要的是當時的掛牌價乘當時的股數，
    用還原價會錯得很離譜：NVDA 2021-07-08 的還原收盤是 19.81，實際掛牌價是 792。

**分割必須另外補，這是本模組最容易錯的地方。**
封面頁股數一季才更新一次，分割卻是當天生效。NVDA 在 2024-06-10 分割 10:1，
而它上一次申報股數是 2024-05-29 的 24.6 億（分割前），下一次要等到 2024-08-28
的 245.3 億。中間那兩個半月若直接用 24.6 億乘分割後的股價，市值會少算十倍 ——
NVDA 會整個掉出前 50，而那正是這段期間漲最多的一檔。
所以股數一律再乘上「股數基準日之後、評價日之前」發生的分割倍數。

**殘存的偏差：候選池本身。**
`collect/us.py` 的 `CANDIDATES` 是今天的大型股清單。一家 2024 年在前 50、
今天已經掉出候選池的公司，這裡還原不出來 —— 這與台股用 PCF 籃子不同，
不是零偏差，只是把偏差從「前 50」縮小到「候選池 80 檔」。
被併購下市的公司更是連 SEC 的現行代號對照表都查不到。
**所以美股的 PIT 快照比台股弱一級，`method` 欄位據實寫明，不要當成同一種東西。**
"""
from __future__ import annotations
import datetime as dt, json, sys, time
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW, CONFIG
from collect import sec_edgar as SE
from collect.us import CANDIDATES, ETFS

# 股數的備援鏈。多重股權級別的公司（GOOGL 的 A/B/C、META 的 A/B）封面頁
# 只按級別揭露，companyfacts 只收沒有維度的 fact，所以 dei 那個標籤整個不存在 ——
# 實測 GOOGL／META／PLTR 三檔都抓不到，而它們是前十大。少了它們，
# 排名會把兩檔本來排不進去的公司推進前 50。
#
# 各層的實測誤差（對 2026-09-18 的 yfinance 市值）：
#   dei 封面頁股數        　基準，77/80 檔有
#   資產負債表流通股數    　GOOGL +0.0%、PLTR −0.0%
#   加權平均稀釋股數      　META +0.7%、但 PLTR +6.9%（重度股權獎酬會高估）
# 所以順序不能顛倒：時點股數優先，期間平均只當最後手段。
SHARE_TAGS = [
    ("dei", "EntityCommonStockSharesOutstanding", True, "封面頁"),
    ("us-gaap", "CommonStockSharesOutstanding", True, "資產負債表"),
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding", False, "加權稀釋"),
]

# ── 股數的單位校正 ────────────────────────────────────────────────
# companyfacts 的股數有兩種單位對不上報價的情形，兩種都不會報錯：
#
#   存託憑證　SEC 申報的是當地普通股，報價是 ADS。TSM 一股 ADS ＝ 5 股台股，
#             不換算的話市值算成 4.76 倍（實測 +375.9%），這檔會永遠排第一。
#   多重股權　封面頁只按級別揭露，而沒有維度的那一筆常常只是**其中一個級別**。
#             實測 BRK-B 拿到的是 94 萬股（A 股），市值算成 4.7 億而不是 1.09 兆；
#             MA −86%、V −75%。比抓不到更糟：有數字、會排序、不會報錯，
#             只是讓三檔兆元級公司安靜地掉出前 50。
#
# 處理方式是**一次性的單位校正**，不是水準調整：
#   k = （某參考日的市值 ÷ 當日收盤）÷ 同日的 SEC 股數
# k 代表「SEC 那個數字要乘多少才是報價單位的總股數」——
# 它是級別結構與 ADS 比例的函數，不是未來資訊。股數的**變動**
# （買回、增發）仍然完全來自 PIT 資料，只有單位是校正過的。
#
# 只在 |k−1| > 10% 時才套用，避免把報價時點差造成的雜訊當成校正。
# k 連同參考日存在 config/us_share_scale.json，可以事後查。
# **前提是級別結構在期間內沒變過** —— 變過的話這個常數會是錯的，
# 而 build() 結尾的對帳只驗得到最新一天。這是本模組已知的弱點。
SCALE_MIN_DEV = 0.10
STALE_DAYS = 5          # 未還原報價的快取容許天數（含週末與假日）
SCALE_FILE = CONFIG / "us_share_scale.json"

SHARES = RAW / "sec_shares"
RAW_PX = RAW / "us" / "prices_raw.parquet"
SPLITS = RAW / "us" / "splits.json"
OUT = CONFIG / "universe_us"
TOP_N = 50


def fetch_shares(codes: list[str], refresh: bool = False) -> dict[str, int]:
    """封面頁流通股數，逐檔快取。回傳各檔筆數。"""
    SHARES.mkdir(parents=True, exist_ok=True)
    tm = SE.ticker_map()
    stat = {}
    for c in codes:
        f = SHARES / f"{c}.json"
        if f.exists() and not refresh:
            stat[c] = len(json.loads(f.read_text(encoding="utf-8"))["shares"])
            continue
        cik = tm.get(c.upper()) or tm.get(c.upper().replace("-", "."))
        if not cik:
            print(f"  {c} 查無 CIK"); stat[c] = 0; continue
        cf = SE._get(SE.FACTS_URL.format(cik=cik))
        time.sleep(SE.SLEEP)
        if not cf:
            print(f"  {c} 無 companyfacts"); stat[c] = 0; continue
        rows, basis = [], ""
        for tax, tag, instant, label in SHARE_TAGS:
            got, _t = SE._collect(cf.get("facts", {}).get(tax, {}), [tag],
                                  instant=instant)
            if not got:
                continue
            if instant:
                rows = [{"end": e, "shares": v["val"], "filed": v["filed"]}
                        for e, v in got.items()]
            else:
                # 期間值：取單季那一筆，期末日當作股數的基準日
                rows = [{"end": k[1], "shares": v["val"], "filed": v["filed"]}
                        for k, v in got.items() if SE._span(*k) == 1]
            if rows:
                basis = label
                break
        if not rows:
            print(f"  {c} 三種股數標籤都沒有"); stat[c] = 0; continue
        rows.sort(key=lambda r: (r["filed"], r["end"]))
        f.write_text(json.dumps({"code": c, "cik": cik, "shares_basis": basis,
                                 "shares": rows}, ensure_ascii=False),
                     encoding="utf-8")
        stat[c] = len(rows)
    return stat


def fetch_prices(codes: list[str], start: str = "2021-07-01",
                 refresh: bool = False) -> pd.DataFrame:
    """**未還原**收盤價與分割紀錄。市值算的是掛牌價 × 掛牌股數。

    快取超過 `STALE_DAYS` 天就自動重抓。少了這一步，每月的新快照會永遠
    建不出來 —— 而症狀是「快照數停在 26 份」，沒有任何錯誤訊息。
    """
    if RAW_PX.exists() and not refresh:
        cached = pd.read_parquet(RAW_PX)
        age = (pd.Timestamp.now().normalize() - cached["date"].max()).days
        if age <= STALE_DAYS:
            return cached
        print(f"  未還原報價已 {age} 天未更新，重抓")
    import yfinance as yf
    df = yf.download(codes, start=start, auto_adjust=False, actions=True,
                     progress=False, group_by="ticker", threads=True)
    rows, splits = [], {}
    for c in codes:
        try:
            d = df[c].dropna(subset=["Close"])
        except Exception:  # noqa: BLE001
            continue
        rows.append(pd.DataFrame({"date": d.index, "code": c,
                                  "close": d["Close"].to_numpy()}))
        if "Stock Splits" in d.columns:
            sp = d["Stock Splits"]
            splits[c] = {str(pd.Timestamp(i).date()): float(v)
                         for i, v in sp[sp > 0].items()}
    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    RAW_PX.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(RAW_PX, index=False)
    SPLITS.write_text(json.dumps(splits, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def build_share_scale(codes: list[str], refresh: bool = False) -> dict:
    """算出每檔的股數單位校正係數 k，存檔。"""
    if SCALE_FILE.exists() and not refresh:
        return json.loads(SCALE_FILE.read_text(encoding="utf-8"))
    import yfinance as yf
    px = fetch_prices(codes)
    last = px["date"].max()
    close = px[px["date"] == last].set_index("code")["close"].to_dict()
    sh, _basis = _load_shares()
    splits = json.loads(SPLITS.read_text(encoding="utf-8")) if SPLITS.exists() else {}
    ds = str(last.date())
    scale = {}
    for c in codes:
        n = pit_shares(sh.get(c) or [], splits.get(c, {}), ds)
        if not n or c not in close:
            continue
        try:
            mcap = yf.Ticker(c).info.get("marketCap")
        except Exception:  # noqa: BLE001
            mcap = None
        time.sleep(0.1)
        if not mcap:
            continue
        k = (float(mcap) / float(close[c])) / n
        if abs(k - 1) > SCALE_MIN_DEV:
            scale[c] = round(k, 6)
    out = {"ref_date": ds, "min_dev": SCALE_MIN_DEV, "scale": scale}
    SCALE_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                          encoding="utf-8")
    return out


def _load_shares() -> tuple[dict[str, list[dict]], dict[str, str]]:
    rows, basis = {}, {}
    for f in sorted(SHARES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows[f.stem] = d["shares"]
        basis[f.stem] = d.get("shares_basis", "?")
    return rows, basis


def pit_shares(rows: list[dict], splits: dict[str, float], d: str,
               scale: float = 1.0) -> float | None:
    """`d` 當天看得到的流通股數，已補分割、已校正成報價單位。"""
    seen = [r for r in rows if r["filed"] <= d]
    if not seen:
        return None
    r = seen[-1]                                  # rows 已按 filed 排序
    n = float(r["shares"])
    for sd, ratio in splits.items():
        if r["end"] < sd <= d:
            n *= ratio
    return n * scale


def build(start: str = "2024-08-01", end: str | None = None,
          top_n: int = TOP_N) -> dict:
    end = end or dt.date.today().isoformat()
    codes = list(dict.fromkeys(CANDIDATES))
    px = fetch_prices(codes)
    sh, sh_basis = _load_shares()
    splits = json.loads(SPLITS.read_text(encoding="utf-8")) if SPLITS.exists() else {}
    scale = build_share_scale(codes)["scale"]
    if scale:
        print(f"股數單位校正 {len(scale)} 檔："
              + "　".join(f"{k}×{v:.3f}" for k, v in sorted(scale.items())))
    OUT.mkdir(parents=True, exist_ok=True)

    dates = []
    d0 = dt.date.fromisoformat(start).replace(day=15)
    dend = dt.date.fromisoformat(end)
    while d0 <= dend:
        dates.append(d0)
        d0 = (d0.replace(day=28) + dt.timedelta(days=7)).replace(day=15)

    px = px.sort_values("date")
    snaps, ever = {}, set()
    for i, day in enumerate(dates, 1):
        # 15 日逢假日就往前找最近的交易日（最多回推 6 天），與台股同一條規則
        win = px[px["date"] <= pd.Timestamp(day)]
        if win.empty:
            continue
        used = win["date"].max()
        if (pd.Timestamp(day) - used).days > 6:
            print(f"  [{i}/{len(dates)}] {day}: 往前 7 天無報價"); continue
        day_px = win[win["date"] == used].set_index("code")["close"].to_dict()
        ds = str(used.date())

        rows = []
        for c, close in day_px.items():
            n = pit_shares(sh.get(c) or [], splits.get(c, {}), ds,
                           scale.get(c, 1.0))
            if n:
                rows.append({"code": c, "mcap": float(close) * n})
        if len(rows) < top_n:
            print(f"  [{i}/{len(dates)}] {ds}: 只有 {len(rows)} 檔算得出市值，跳過")
            continue
        rows.sort(key=lambda r: r["mcap"], reverse=True)
        top = rows[:top_n]
        total = sum(r["mcap"] for r in top) or 1.0
        cons = [{"code": r["code"], "name": r["code"], "industry": "",
                 "mcap": r["mcap"], "weight": r["mcap"] / total} for r in top]
        # ETF 固定納入（與 collect/us.py 同一條），但歷史規模不還原 ——
        # 它們不參與市值排序，硬編一個數字只會讓 weight 看起來有根據。
        cons += [{"code": t, "name": t, "industry": "ETF",
                  "mcap": None, "weight": None} for t in ETFS]

        stamp = used.strftime("%Y%m%d")
        uni = {"as_of": stamp, "built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
               "market": "US", "method": "sec_pit_shares_x_unadjusted_close",
               "candidate_pool": len(codes), "size": len(cons),
               "shares_basis": {c["code"]: sh_basis.get(c["code"], "?")
                                for c in cons if c["industry"] != "ETF"},
               "note": ("候選池是今天的大型股清單，已掉出池外或下市者還原不了；"
                        "這份快照比台股 PCF 弱一級"),
               "constituents": cons}
        (OUT / f"{stamp}.json").write_text(
            json.dumps(uni, ensure_ascii=False, indent=2), encoding="utf-8")
        snaps[stamp] = [c["code"] for c in cons]
        ever |= set(snaps[stamp])
        print(f"  [{i}/{len(dates)}] {ds}: {len(top)} 檔 ＋ {len(ETFS)} ETF")
        
    meta = {"built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "source": "SEC dei:EntityCommonStockSharesOutstanding × yfinance 未還原收盤",
            "snapshots": len(snaps), "codes_ever": sorted(ever), "dates": sorted(snaps)}
    print(f"\n快照 {len(snaps)} 份｜曾入選過 {len(ever)} 檔")
    _reconcile(snaps)
    return meta


RECON_TOL = 0.25


def recon_diff(constituents: list[dict], live: dict[str, float],
               tol: float = RECON_TOL) -> list[str]:
    """重建 vs 即時，兩側都查。

    只查「重建榜上的人」是不夠的 —— BRK-B／MA／V 的股數被算成十分之一，
    直接掉出前 50，於是那份對帳一句話都沒說。
    一個只驗得到自己選中的人的對帳，正好漏掉最嚴重的那一種錯。
    """
    bad = []
    for c in constituents:
        m, ref = c.get("mcap"), live.get(c["code"])
        if not m or not ref:
            continue
        if abs(m / ref - 1) > tol:
            bad.append(f"{c['code']} {m / ref - 1:+.0%}")
    for c in sorted(set(live) - {c["code"] for c in constituents}):
        bad.append(f"{c}(即時在榜、重建沒有)")
    return bad


def _reconcile(snaps: dict) -> list[str]:
    """拿最新一份重建快照對帳 yfinance 的即時市值。

    這道對帳是給**未來**用的：今天只有 TSM 一檔 ADR，而它已經列在
    `ADS_RATIO` 裡。但候選池會換人，下一檔 ADR 進來時沒有人會想起這件事 ——
    而它的症狀是「永遠排第一」，不是「報錯」。誤差超過 25% 就叫出來。
    """
    live_p = CONFIG / "universe_us_latest.json"
    if not snaps or not live_p.exists():
        return []
    live = {c["code"]: c["mcap"] for c in
            json.loads(live_p.read_text(encoding="utf-8"))["constituents"]
            if c.get("mcap")}
    newest = json.loads((OUT / f"{sorted(snaps)[-1]}.json").read_text(encoding="utf-8"))
    bad = recon_diff(newest["constituents"], live)
    if bad:
        print(f"⚠ 與即時市值差超過 {RECON_TOL:.0%} 的有 {len(bad)} 檔：{'　'.join(bad)}")
        print("  最可能的原因是存託憑證的股數單位 —— 檢查 ADS_RATIO。")
    else:
        print(f"對帳：最新快照的市值與即時值全部落在 ±{RECON_TOL:.0%} 內")
    return bad


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    codes = list(dict.fromkeys(CANDIDATES))
    st = fetch_shares(codes, refresh="--refresh-shares" in sys.argv)
    ok = sum(1 for v in st.values() if v)
    print(f"封面頁股數：{ok}/{len(codes)} 檔　"
          f"（無資料：{' '.join(k for k, v in st.items() if not v) or '無'}）\n")
    build(args[0] if args else "2024-08-01",
          args[1] if len(args) > 1 else None)
