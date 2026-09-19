# -*- coding: utf-8 -*-
"""美股一年期判斷用的基本面特徵。季頻，依法定申報期限對齊到可用日。

與台股那支（features/fundamentals.py）刻意做成同一個介面
（FUND_COLS／build()／as_of()），因為檢查點、滾動、規則模型三個地方
都要按市場切換來源；介面不同就得在三個地方各寫一次 if。

可用日用法定申報期限，理由與台股同一條：
  季末日當可用日＝前視偏差，而且它不會讓回測失敗、只會讓回測變漂亮。

美國期限（大型加速申報公司，本清單全部符合）：
  10-Q 季末後 40 天　·　10-K 會計年度結束後 60 天
外國發行人（本清單只有 TSM）不適用 10-Q，實際走本國規則：
  季末後 45 天　·　年度結束後 90 天（與台股 Q4 → 次年 3/31 一致）

**這裡的可用日是法定上限，不是實際申報日。**
實際上多數公司提早 2–3 週公布，所以我們會比真實世界晚知道前提翻掉。
要拿到真正的申報日得走 SEC EDGAR 的 XBRL companyfacts（欄位 `filed`），
那需要在 User-Agent 放一個聯絡信箱才不會被擋，未取得同意前不做。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW

SRC = RAW / "us_fund"
FPI = {"TSM"}                     # 外國發行人：走本國期限，不是 10-Q
LAG_Q, LAG_FY = 40, 60            # 天
LAG_Q_FPI, LAG_FY_FPI = 45, 90

# 與台股同名同義 —— 名字一樣時，定義也必須一樣，否則橫斷面比較會靜默失真。
# 唯一的例外是 eps_yoy：季數不足 8 季時退回「單季年增」，
# 退回與否逐列記在 eps_yoy_basis，不藏起來。
FUND_COLS = ["gross_margin", "op_margin", "gm_chg_4q", "eps_ttm", "eps_yoy",
             "roe_ttm", "rev_cagr_3y", "fcf_margin", "capex_intensity",
             "debt_ratio", "rev_yoy", "rev_yoy_ttm"]

# 淨利相對營業利益的倍數。超過這個倍數就代表當季獲利主要來自非營業項目
# （處分利益、投資評價、稅務一次性），拿它算出來的 EPS 年增不是獲利能力。
# 不作廢那個數字（它是真的），但逐列標記，讓論點文字說得出來。
NONOP_MULT = 2.0
US_TAX = 0.21     # 聯邦法定稅率。用固定值而非實際有效稅率，
                  # 是為了讓「營業利益本益比」在橫斷面上可比 ——
                  # 各公司的有效稅率差異本身就是一次性項目的產物。


def _avail(period_end: pd.Timestamp, is_fy: bool, code: str) -> pd.Timestamp:
    if code in FPI:
        return period_end + pd.Timedelta(days=LAG_FY_FPI if is_fy else LAG_Q_FPI)
    return period_end + pd.Timedelta(days=LAG_FY if is_fy else LAG_Q)


def _load_one(f: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = json.loads(f.read_text(encoding="utf-8"))
    q = pd.DataFrame(list((d.get("quarters") or {}).values()))
    a = pd.DataFrame(list((d.get("annual") or {}).values()))
    for x in (q, a):
        if not x.empty:
            x["code"] = d["code"]
            x["date"] = pd.to_datetime(x["period_end"])
    return q, a


def _safe_ratio(num: pd.Series, den: pd.Series, lo: float, hi: float,
                min_den_frac: float = 0.2) -> pd.Series:
    """比率，但分母接近零時作廢。

    台股那邊實測過：分母趨近零會算出毛利率 251%、存貨天數 34,197 天，
    數字看起來有根據、實際上是除法的產物。判準用「相對這家公司自己的
    中位數」而不是絕對門檻 —— 各公司的規模差三個數量級。
    """
    d = den.astype(float)
    med = d.abs().median()
    ok = d.abs() > (min_den_frac * med if pd.notna(med) and med > 0 else 0)
    r = pd.Series(np.where(ok, num.astype(float) / d.replace(0, np.nan), np.nan),
                  index=num.index)
    return r.where(r.between(lo, hi))


_CACHE: dict[str, pd.DataFrame] = {}


def build(refresh: bool = False) -> pd.DataFrame:
    """回傳 [code, date, avail_date, *FUND_COLS, eps_yoy_basis]。"""
    if not SRC.exists():
        return pd.DataFrame(columns=["code", "date", "avail_date", *FUND_COLS])
    stamp = str(sorted((f.name, int(f.stat().st_mtime)) for f in SRC.glob("*.json")))
    if not refresh and stamp in _CACHE:
        return _CACHE[stamp].copy()
    out = _build_uncached()
    _CACHE.clear()
    _CACHE[stamp] = out
    return out.copy()


def _build_uncached() -> pd.DataFrame:
    frames = []
    for f in sorted(SRC.glob("*.json")):
        q, a = _load_one(f)
        if q.empty or "revenue" not in q.columns:
            continue
        q = q.sort_values("date").reset_index(drop=True)
        code = q["code"].iloc[0]
        fy_ends = set(a["period_end"]) if not a.empty else set()

        for c in ("gross", "op_income", "net_income", "eps", "capex", "cfo",
                  "equity", "assets"):
            if c not in q.columns:
                q[c] = np.nan

        # yfinance 的季度現金流已是單季值（非年初至今），實測 NVDA／TSM
        # 皆為 3 個月期間。這裡仍加一道守恆檢查（見 __main__），
        # 若哪天上游改成累計制，四季加總會對不上年報而被抓出來。
        q["capex"] = q["capex"].abs()

        r = q["revenue"].astype(float)
        q["rev_ttm"] = r.rolling(4).sum()
        q["ni_ttm"] = q["net_income"].astype(float).rolling(4).sum()
        q["eps_ttm"] = q["eps"].astype(float).rolling(4).sum()
        q["capex_ttm"] = q["capex"].astype(float).rolling(4).sum()
        q["cfo_ttm"] = q["cfo"].astype(float).rolling(4).sum()

        q["gross_margin"] = _safe_ratio(q["gross"], r, -1.0, 1.0)
        q["op_margin"] = _safe_ratio(q["op_income"], r, -5.0, 2.0)
        q["gm_chg_4q"] = q["gross_margin"] - q["gross_margin"].shift(4)
        q["rev_yoy"] = _safe_ratio(r - r.shift(4), r.shift(4), -1.0, 20.0)
        q["rev_yoy_ttm"] = _safe_ratio(q["rev_ttm"] - q["rev_ttm"].shift(4),
                                       q["rev_ttm"].shift(4), -1.0, 20.0)
        q["capex_intensity"] = _safe_ratio(q["capex_ttm"], q["rev_ttm"], 0.0, 3.0)
        q["fcf_margin"] = _safe_ratio(q["cfo_ttm"] - q["capex_ttm"].fillna(0),
                                      q["rev_ttm"], -5.0, 2.0)
        eq = q["equity"].astype(float)
        # 平均股東權益為負時 ROE 一律作廢。
        # ABBV 的權益是 −59.4 億（長年大額買回＋分拆），照算會得到 −206%，
        # 而它其實是穩定獲利的公司 —— 負分母會把訊號的正負號整個翻過來，
        # 在橫斷面百分位裡直接把最賺錢的一批排到最後。
        avg_eq = (eq + eq.shift(4)) / 2
        avg_eq = avg_eq.where(avg_eq > 0)
        q["roe_ttm"] = _safe_ratio(q["ni_ttm"], avg_eq, -3.0, 3.0)
        q["debt_ratio"] = 1 - _safe_ratio(eq, q["assets"].astype(float), 0.0, 1.0)

        # EPS 年增：有 8 季就用 TTM 比 TTM（濾掉季節性），
        # 不足就退回單季年增，並把用了哪一種記在欄位裡。
        ttm_ok = q["eps_ttm"].notna() & q["eps_ttm"].shift(4).notna()
        yoy_ttm = _safe_ratio(q["eps_ttm"] - q["eps_ttm"].shift(4),
                              q["eps_ttm"].shift(4), -5.0, 10.0)
        yoy_q = _safe_ratio(q["eps"].astype(float) - q["eps"].astype(float).shift(4),
                            q["eps"].astype(float).shift(4), -5.0, 10.0)
        q["eps_yoy"] = yoy_ttm.where(ttm_ok, yoy_q)
        q["eps_yoy_basis"] = np.where(ttm_ok, "TTM年增", "單季年增")
        # 實測 GOOGL 2026Q2：營業利益 407.7 億、淨利 1,121.9 億 —— 淨利是
        # 營業利益的 2.75 倍，EPS 因此年增 +294%。那個數字沒錯，但它不是
        # 本業變好，而且一年期判斷最常犯的錯就是把一次性利益當成獲利軌跡。
        ratio = (q["net_income"].astype(float).abs()
                 / q["op_income"].astype(float).abs().replace(0, np.nan))
        q["nonop_heavy"] = (ratio > NONOP_MULT).fillna(False)
        # 營業利益版的每股盈餘，給「本益比被一次性利益灌水」的公司用。
        # 股數由淨利／EPS 反推 —— 兩者同樣被一次性利益放大，相除後抵銷，
        # 所以這個股數仍然是對的。
        shares = (q["net_income"].astype(float)
                  / q["eps"].astype(float).replace(0, np.nan))
        q["op_income_ttm"] = q["op_income"].astype(float).rolling(4).sum()
        q["eps_op_ttm"] = q["op_income_ttm"] * (1 - US_TAX) / shares

        # 三年營收 CAGR 走年報。年報的可用日是該年度的 10-K 期限，
        # 不能沿用季報的可用日 —— 那會讓三年成長率提早三個月出現。
        q["rev_cagr_3y"] = np.nan
        if not a.empty and "revenue" in a.columns:
            av = (a.dropna(subset=["revenue"]).sort_values("date")
                   .assign(av=lambda x: [
                       _avail(pd.Timestamp(d), True, code) for d in x["date"]]))
            if len(av) >= 4:
                for i in range(3, len(av)):
                    v0, v1 = float(av["revenue"].iloc[i - 3]), float(av["revenue"].iloc[i])
                    if v0 > 0 and v1 > 0:
                        g = (v1 / v0) ** (1 / 3) - 1
                        m = q["date"] >= av["date"].iloc[i]
                        q.loc[m & (q["date"] < (av["date"].iloc[i + 1]
                                                if i + 1 < len(av)
                                                else pd.Timestamp("2100-01-01"))),
                              "rev_cagr_3y"] = g

        q["avail_date"] = [
            _avail(d, str(pd.Timestamp(d).date()) in fy_ends, code) for d in q["date"]]
        frames.append(q[["code", "date", "avail_date", *FUND_COLS,
                         "eps_yoy_basis", "nonop_heavy", "eps_op_ttm", "revenue"]])

    if not frames:
        return pd.DataFrame(columns=["code", "date", "avail_date", *FUND_COLS])
    return pd.concat(frames, ignore_index=True).sort_values(["code", "date"])


def as_of(d: pd.Timestamp | str) -> pd.DataFrame:
    """取 d 當天「已申報」的最新一季。與台股同一條規則：
    嚴格用 avail_date 篩，整列皆空的季別跳過，並用 drop_duplicates 取最後一列
    （groupby().last() 是逐欄取最後非空值，會把不同季的數字拼成同一列）。"""
    d = pd.Timestamp(d)
    f = build()
    if f.empty:
        return f
    f = f[f["avail_date"] <= d]
    # 判準是「這一季有沒有營收」，不是「有沒有任何一欄不是空的」。
    # 實測 AMZN 2026Q2 在上游只有 EPS 一欄，其餘全空 —— 用 any() 會選中它，
    # 於是 AMZN 的毛利率、ROE、資本支出強度全部變成缺值，
    # 而它 2026Q1 那一季其實是完整的。沒有營收就算不出任何比率，
    # 那一列不是「資料較少的一季」，是「還沒有的一季」。
    f = f[f["revenue"].notna() & f[FUND_COLS].notna().any(axis=1)]
    if f.empty:
        return f
    return (f.sort_values(["code", "date"])
             .drop_duplicates("code", keep="last").reset_index(drop=True))


if __name__ == "__main__":
    f = build()
    print(f"美股基本面：{len(f):,} 列 × {f['code'].nunique()} 檔　"
          f"季別 {f['date'].min().date()} → {f['date'].max().date()}")
    cov = f[FUND_COLS].notna().mean().sort_values(ascending=False)
    print("\n覆蓋率：")
    for k, v in cov.items():
        print(f"  {k:<18}{v*100:>5.1f}%")
    print("\n守恆檢查（四季加總 vs 年報，偏離 >3% 才列出）：")
    bad = 0
    for p in sorted(SRC.glob("*.json")):
        q, a = _load_one(p)
        if q.empty or a.empty or "revenue" not in q.columns or "revenue" not in a.columns:
            continue
        a = a.dropna(subset=["revenue"]).sort_values("date")
        for _, ar in a.iterrows():
            w = q[(q["date"] > ar["date"] - pd.Timedelta(days=370))
                  & (q["date"] <= ar["date"])].dropna(subset=["revenue"])
            if len(w) != 4:
                continue
            s, t = w["revenue"].sum(), float(ar["revenue"])
            if t and abs(s / t - 1) > 0.03:
                print(f"  {q['code'].iloc[0]:<6}{str(ar['date'].date())}　"
                      f"四季 {s/1e9:,.1f} vs 年報 {t/1e9:,.1f}　"
                      f"({s/t-1:+.1%})")
                bad += 1
    print(f"  共 {bad} 筆不符")
