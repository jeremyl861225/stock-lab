"""一年期論點的檢查點：把「一年後會漲」拆成幾個幾週內就能證偽的主張。

為什麼需要這個：
  h=250 的預測要 2027 年才結算，而這套系統的價值全在回饋迴路。
  更糟的是日頻預測在 250 日尺度上重疊 249/250，兩年資料每檔只有約
  2 筆獨立觀測 —— 等一年拿到的也不是統計，是軼事。

  所以一年期判斷不是「押一個方向然後等」，而是要寫清楚：
  這個論點成立的前提是什麼？哪個數字翻掉就代表我錯了？
  前提每季（月營收每月）自動對帳，論點壞掉當下就知道，不必等到期。

  這也治一個更隱蔽的毛病：說不出前提的看多，通常不是判斷，是氛圍。

檢查點狀態：
  holding  前提成立，論點續存
  broken   前提已被資料推翻 —— 不等到期就該重寫判斷
  pending  該期資料還沒公告
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RAW
from features import fundamentals as F

OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
       ">": lambda a, b: a > b, "<": lambda a, b: a < b}


def _monthly_rev() -> pd.DataFrame:
    """月營收：YoY 與 TTM YoY。月營收每月 10 日前公告，是最快的前提對帳來源。"""
    d = RAW / "finmind" / "rev"
    rows = []
    for f in d.glob("*.json"):
        rows += json.loads(f.read_text(encoding="utf-8")).get("payload", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).rename(columns={"stock_id": "code"})
    g = df.groupby("code")["revenue"]
    df["rev_yoy"] = g.transform(lambda s: s / s.shift(12) - 1)
    ttm = g.transform(lambda s: s.rolling(12).sum())
    df["rev_yoy_ttm"] = df.assign(t=ttm).groupby("code")["t"].transform(
        lambda s: s / s.shift(12) - 1)
    # 法定公告日：次月 10 日。用它當可用日，不用資料月份。
    df["avail_date"] = (df["date"] + pd.offsets.MonthBegin(1)
                        + pd.Timedelta(days=9))
    return df[["code", "date", "avail_date", "rev_yoy", "rev_yoy_ttm"]]


def _metric_series(code: str, metric: str) -> pd.DataFrame:
    """回傳 [avail_date, value]，已按可用日對齊（不含未公告的期別）。"""
    if metric in ("rev_yoy", "rev_yoy_ttm"):
        m = _monthly_rev()
        if m.empty:
            return pd.DataFrame()
        m = m[m["code"] == code]
        return m[["avail_date", metric]].rename(columns={metric: "value"}).dropna()
    f = F.build()
    if f.empty or metric not in f.columns:
        return pd.DataFrame()
    f = f[f["code"] == code]
    return f[["avail_date", metric]].rename(columns={metric: "value"}).dropna()


def score(cp: dict, code: str, today: pd.Timestamp | str) -> dict:
    """評一個檢查點。只用 today 當天已公告的資料。"""
    today = pd.Timestamp(today)
    s = _metric_series(code, cp["metric"])
    s = s[s["avail_date"] <= today] if not s.empty else s
    if s.empty:
        return {**cp, "status": "pending", "actual": None, "as_of": None}
    latest = s.sort_values("avail_date").iloc[-1]
    ok = OPS[cp["op"]](float(latest["value"]), float(cp["threshold"]))
    return {**cp, "status": "holding" if ok else "broken",
            "actual": round(float(latest["value"]), 4),
            "as_of": str(pd.Timestamp(latest["avail_date"]).date())}


def score_all(judgment: dict, today: pd.Timestamp | str) -> dict:
    """評一檔的全部檢查點，回傳彙總。"""
    cps = judgment.get("checkpoints") or []
    res = [score(c, judgment["code"], today) for c in cps]
    n_b = sum(r["status"] == "broken" for r in res)
    n_h = sum(r["status"] == "holding" for r in res)
    n_e = n_b + n_h
    # 判定：過半前提被推翻 → 論點失效，該重寫而不是等到期
    verdict = ("失效" if n_e and n_b * 2 > n_e else
               "成立" if n_e and n_b == 0 else
               "待驗" if n_e == 0 else "動搖")
    return {"code": judgment["code"], "verdict": verdict,
            "holding": n_h, "broken": n_b, "pending": len(res) - n_e,
            "checkpoints": res}


if __name__ == "__main__":
    import glob
    fs = sorted(glob.glob(str(Path(__file__).parent.parent / "judgments" / "*_1y*.json")))
    if not fs:
        print("尚無一年期判斷檔（judgments/*_1y*.json）")
        sys.exit(0)
    today = pd.Timestamp.today()
    for j in json.load(open(fs[-1]))["judgments"]:
        r = score_all(j, today)
        print(f"{r['code']:<6}{r['verdict']:<5}成立{r['holding']} 推翻{r['broken']} 待公告{r['pending']}")
        for c in r["checkpoints"]:
            mark = {"holding": "✓", "broken": "✗", "pending": "·"}[c["status"]]
            act = "—" if c["actual"] is None else f"{c['actual']:+.3f}"
            print(f"   {mark} {c['claim']}  實際 {act}")
