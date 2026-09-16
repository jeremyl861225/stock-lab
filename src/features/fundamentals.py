"""一年期判斷用的基本面特徵。季頻，依法定公告期限對齊到可用日。

為什麼不能用季末日當可用日：
  2026-06-30 的財報，法定公告期限是 2026-08-14。若拿 6/30 當可用日，
  等於在 7 月就用到 8 月中才公開的數字 —— 這是最典型的前視偏差，
  而且它不會讓回測失敗、只會讓回測變漂亮，所以特別危險。
  這裡一律用法定上限日，寧可晚用也不早用（實際公告多半更早，
  用上限日是保守側，不會高估可得性）。

台灣法定期限（證交法 §36）：
  Q1 3/31 → 5/15 ／ Q2 6/30 → 8/14 ／ Q3 9/30 → 11/14 ／ Q4 12/31 → 次年 3/31
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW

# 季末 (月, 日) → 法定公告上限 (月偏移, 月, 日)
PUB = {(3, 31): (0, 5, 15), (6, 30): (0, 8, 14),
       (9, 30): (0, 11, 14), (12, 31): (1, 3, 31)}

FUND_COLS = ["gross_margin", "op_margin", "gm_chg_4q", "eps_ttm", "eps_yoy",
             "roe_ttm", "rev_cagr_3y", "fcf_margin", "capex_intensity",
             "debt_ratio", "inv_days_chg"]


def _pub_date(q: pd.Timestamp) -> pd.Timestamp:
    """季末日 → 法定公告上限日。非標準季末（少數公司）退回加 45 天。"""
    key = (q.month, q.day)
    if key not in PUB:
        return q + pd.Timedelta(days=45)
    yr_off, m, d = PUB[key]
    return pd.Timestamp(year=q.year + yr_off, month=m, day=d)


def _load(kind: str) -> pd.DataFrame:
    d = RAW / "finmind" / kind
    if not d.exists():
        return pd.DataFrame()
    rows = []
    for f in d.glob("*.json"):
        for r in json.loads(f.read_text(encoding="utf-8")).get("payload", []):
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.pivot_table(index=["stock_id", "date"], columns="type",
                        values="value", aggfunc="last").reset_index()
    df.columns.name = None
    return df.rename(columns={"stock_id": "code"})


QMON = {(3, 31): ("01", "02", "03"), (6, 30): ("04", "05", "06"),
        (9, 30): ("07", "08", "09"), (12, 31): ("10", "11", "12")}


def _reconcile() -> set[tuple[str, str]]:
    """季報營收 vs 同期月營收加總，回傳對不起來的 (code, 季末日)。

    為什麼需要：實測 2059（川湖）2026Q2 的季報營收是月營收加總的 1.30 倍，
    而它其他六季都穩定在 1.00–1.07 —— 那一筆壞了。毛利率因此從常態的
    76% 跳到 87%，EPS 跳成 74.38（單季兩倍）。這種錯不會讓程式當掉，
    只會讓判斷建立在假數字上。

    判準用「相對該檔自己的中位數」而不是絕對 1.0：合併報表涵蓋子公司，
    月營收揭露有的是母公司單體，比值本來就可能穩定偏離 1.0。
    偏離自身常態才是訊號。

    金控除外：金融業的「營業收入」與月營收揭露基礎不同，本來就不可比，
    拿來對帳只會產生大量假警報（實測 151 個異常裡最嚴重的六個全是金控）。
    """
    import json as _json
    mrev: dict[str, dict[str, float]] = {}
    for f in (RAW / "finmind" / "rev").glob("*.json"):
        for x in _json.loads(f.read_text(encoding="utf-8")).get("payload", []):
            mrev.setdefault(x["stock_id"], {})[x["date"][:7]] = x.get("revenue")

    bad: set[tuple[str, str]] = set()
    for f in (RAW / "finmind" / "fin").glob("*.json"):
        code = f.stem
        if code.startswith("28"):          # 金控／銀行／保險
            continue
        m = mrev.get(code)
        if not m:
            continue
        qrev = {x["date"]: x["value"] for x in
                _json.loads(f.read_text(encoding="utf-8")).get("payload", [])
                if x["type"] == "Revenue"}
        ratios = {}
        for d, v in qrev.items():
            key = (int(d[5:7]), int(d[8:10]))
            if key not in QMON or not v:
                continue
            ms = [m.get(f"{d[:4]}-{mm}") for mm in QMON[key]]
            if any(x is None for x in ms) or not sum(ms):
                continue
            ratios[d] = v / sum(ms)
        if len(ratios) < 4:                 # 樣本太少，沒有可靠的自身常態
            continue
        med = sorted(ratios.values())[len(ratios) // 2]
        for d, r in ratios.items():
            if med and abs(r / med - 1) > 0.15:
                bad.add((code, d))
    return bad


def build() -> pd.DataFrame:
    """回傳 [code, avail_date, <FUND_COLS>]。avail_date 是「這天起才看得到」。"""
    fin, bs, cf = _load("fin"), _load("bs"), _load("cf")
    if fin.empty:
        return pd.DataFrame(columns=["code", "avail_date", *FUND_COLS])

    df = fin
    for other in (bs, cf):
        if not other.empty:
            df = df.merge(other, on=["code", "date"], how="outer", suffixes=("", "_y"))
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["code", "date"])
    g = df.groupby("code")

    def col(name):
        return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)

    rev, gp, oi = col("Revenue"), col("GrossProfit"), col("OperatingIncome")
    out = pd.DataFrame({"code": df["code"], "date": df["date"]})

    # ── 獲利能力 ──────────────────────────────────────────────
    out["gross_margin"] = gp / rev.replace(0, np.nan)
    out["op_margin"] = oi / rev.replace(0, np.nan)
    # 毛利率四季變化：相對自身一年前，避開產業別的水準差異
    out["gm_chg_4q"] = out.groupby("code")["gross_margin"].transform(lambda s: s - s.shift(4))

    eps = col("EPS")
    out["eps_ttm"] = df.assign(e=eps).groupby("code")["e"].transform(lambda s: s.rolling(4).sum())
    out["eps_yoy"] = out.groupby("code")["eps_ttm"].transform(
        lambda s: s / s.shift(4).abs().replace(0, np.nan) - 1)

    ni = col("IncomeAfterTaxes")
    # 陷阱：EquityAttributableToOwnersOfParent 這個名字在損益表與資產負債表
    # 都有，意思完全不同 —— 損益表那個是「綜合損益歸屬母公司」（流量），
    # 資產負債表那個才是「權益餘額」（存量）。merge 之後同名欄位保留損益表版本，
    # 直接取會算出 ROE 200–600% 這種不可能的數字（實測全部 33 檔都中）。
    # 這裡明確取資產負債表側：merge 的 suffix 是 _y，退路用只存在於
    # 資產負債表的 Equity（權益總額，含非控制權益，會略微低估 ROE，是保守側）。
    eq = col("EquityAttributableToOwnersOfParent_y")
    if eq.isna().all():
        eq = col("Equity")
    ni_ttm = df.assign(x=ni).groupby("code")["x"].transform(lambda s: s.rolling(4).sum())
    # 權益用期初期末平均，避免當期增資把 ROE 壓低成假訊號
    eq_avg = df.assign(x=eq).groupby("code")["x"].transform(lambda s: s.rolling(2).mean())
    out["roe_ttm"] = ni_ttm / eq_avg.replace(0, np.nan)

    # ── 成長 ──────────────────────────────────────────────────
    rev_ttm = df.assign(x=rev).groupby("code")["x"].transform(lambda s: s.rolling(4).sum())
    out["rev_cagr_3y"] = df.assign(x=rev_ttm).groupby("code")["x"].transform(
        lambda s: (s / s.shift(12)) ** (1 / 3) - 1)

    # ── 現金與擴張 ────────────────────────────────────────────
    ocf = col("NetCashInflowFromOperatingActivities")
    capex = col("PropertyAndPlantAndEquipment").abs()
    ocf_ttm = df.assign(x=ocf).groupby("code")["x"].transform(lambda s: s.rolling(4).sum())
    cap_ttm = df.assign(x=capex).groupby("code")["x"].transform(lambda s: s.rolling(4).sum())
    out["fcf_margin"] = (ocf_ttm - cap_ttm) / rev_ttm.replace(0, np.nan)
    # 資本支出強度：一年期判斷裡這是「公司自己押的注」，比任何外部預估都直接
    out["capex_intensity"] = cap_ttm / rev_ttm.replace(0, np.nan)

    # ── 財務結構 ──────────────────────────────────────────────
    out["debt_ratio"] = col("Liabilities") / col("TotalAssets").replace(0, np.nan)
    cogs = col("CostOfGoodsSold").abs()
    inv_days = col("Inventories") / cogs.replace(0, np.nan) * 91.25
    out["inv_days_chg"] = df.assign(x=inv_days).groupby("code")["x"].transform(
        lambda s: s - s.shift(4))

    out["avail_date"] = out["date"].map(_pub_date)
    out = out.replace([np.inf, -np.inf], np.nan)

    # 對不起來的季別整列作廢。寧可少一季，不要拿壞數字做一年期判斷。
    bad = _reconcile()
    if bad:
        key = list(zip(out["code"], out["date"].dt.strftime("%Y-%m-%d")))
        mask = pd.Series([k in bad for k in key], index=out.index)
        out.loc[mask, FUND_COLS] = np.nan
        out.attrs["rejected"] = len(bad)

    # eps_yoy 的基期若接近零，比值會爆成幾百倍（南亞 +1412%、華邦電 +1232%）。
    # 那不是成長，是除以零。基期絕對值小於 0.5 元就不給數字。
    base = out.groupby("code")["eps_ttm"].shift(4)
    out.loc[base.abs() < 0.5, "eps_yoy"] = np.nan

    return out[["code", "date", "avail_date", *FUND_COLS]].reset_index(drop=True)


def as_of(d: pd.Timestamp | str) -> pd.DataFrame:
    """取 d 當天「已公告」的最新一季。嚴格用 avail_date 篩，不用季末日。"""
    d = pd.Timestamp(d)
    f = build()
    if f.empty:
        return f
    f = f[f["avail_date"] <= d]
    return f.sort_values("date").groupby("code", as_index=False).last()


if __name__ == "__main__":
    f = build()
    print(f"基本面特徵：{len(f):,} 列 × {f['code'].nunique()} 檔")
    print(f"季別範圍 {f['date'].min().date()} → {f['date'].max().date()}")
    cov = f[FUND_COLS].notna().mean().sort_values(ascending=False)
    print("\n覆蓋率：")
    for k, v in cov.items():
        print(f"  {k:<18}{v*100:>5.1f}%")
