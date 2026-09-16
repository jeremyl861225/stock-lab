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
    eq = col("EquityAttributableToOwnersOfParent")
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
