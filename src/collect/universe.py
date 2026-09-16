"""建立投資母體（universe）= 市值前 50 大上市公司。

為什麼自己算而不爬元大官網：
  1. 元大頁面為 JS 渲染且會改版，爬蟲是整條管線最脆弱的一環。
  2. 臺灣50指數的選股邏輯本就是市值排序（經流通量調整），自算誤差極小。
  3. 最重要：universe 本身會隨時間變動，回測必須用「當時的」成分股。
     每次產生都帶 as_of 並落盤存檔，才不會犯生存者偏差。
"""
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CONFIG
from collect import twse


def _num(s) -> float:
    if s is None:
        return 0.0
    s = str(s).replace(",", "").strip()
    if s in ("", "--", "-", "X"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def build(date: str, size: int = 50) -> dict:
    """date: YYYYMMDD 交易日。回傳 universe dict 並存檔。"""
    quotes = twse.daily_quotes(date)
    basics = twse.listed_companies()

    # 只留上市普通股：4 碼純數字，且出現在上市公司清單中（排除 ETF/權證/債券）
    listed = {b["code"]: b for b in basics if b["code"].isdigit() and len(b["code"]) == 4}
    rows = []
    for q in quotes:
        code = q["code"]
        if code not in listed:
            continue
        close = _num(q["close"])
        b = listed[code]
        shares = _num(b["shares"])
        if shares <= 0:  # 少數公司股數欄位為空，用實收資本額/面額回補
            par = _num(b["par"]) or 10.0
            shares = _num(b["capital"]) / par if par else 0.0
        mcap = close * shares
        if close <= 0 or mcap <= 0:
            continue
        rows.append({"code": code, "name": b["name"], "industry": b["industry"],
                     "close": close, "shares": shares, "mcap": mcap})

    rows.sort(key=lambda r: r["mcap"], reverse=True)
    top = rows[:size]
    total = sum(r["mcap"] for r in top)

    uni = {
        "as_of": date,
        "built_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "method": "market_cap_top_n_listed_common",
        "size": len(top),
        "constituents": [{
            "code": r["code"], "name": r["name"], "industry": r["industry"],
            "mcap": r["mcap"], "weight": r["mcap"] / total,
        } for r in top],
    }
    out = CONFIG / "universe"
    out.mkdir(exist_ok=True)
    (out / f"{date}.json").write_text(json.dumps(uni, ensure_ascii=False, indent=2), encoding="utf-8")
    (CONFIG / "universe_latest.json").write_text(json.dumps(uni, ensure_ascii=False, indent=2), encoding="utf-8")
    return uni


def load(as_of: str | None = None) -> dict:
    """取用 as_of 當下最接近、且不晚於 as_of 的 universe（point-in-time）。"""
    out = CONFIG / "universe"
    files = sorted(out.glob("*.json")) if out.exists() else []
    if not files:
        raise FileNotFoundError("尚未建立 universe，請先執行 build()")
    if as_of is None:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    ok = [f for f in files if f.stem <= as_of]
    if not ok:
        raise ValueError(f"{as_of} 之前沒有可用的 universe（會造成前視偏誤）")
    return json.loads(ok[-1].read_text(encoding="utf-8"))


if __name__ == "__main__":
    import sys as _s
    d = _s.argv[1] if len(_s.argv) > 1 else dt.datetime.now().strftime("%Y%m%d")
    u = build(d)
    print(f"universe as_of={u['as_of']} size={u['size']}")
    for i, c in enumerate(u["constituents"][:12], 1):
        print(f"  {i:2d}. {c['code']} {c['name']:<8} 市值 {c['mcap']/1e12:6.2f} 兆  權重 {c['weight']*100:5.2f}%")
    print(f"  ... 前 10 大合計權重 {sum(c['weight'] for c in u['constituents'][:10])*100:.2f}%")
