"""橫斷面評估 —— 比方向準確率誠實得多的尺度。

為什麼需要這個：
  回測顯示過去 12 個月台股 20 日上漲基本率高達 67.1%，always_up 因此
  天然佔優，任何模型只要偶爾看空就會扣分。但實務上你不是在賭大盤方向，
  而是在「50 檔裡挑哪幾檔」—— 這時候該問的是：
      模型看好的那批，是否真的漲得比它看壞的那批多？
  這個問題在多頭、空頭、盤整都成立，不會被市場整體方向污染。

指標：
  rank_ic  : 每日 prob_up 與未來報酬的 Spearman 相關（越正越好，0.03 就算不錯）
  ic_ir    : rank_ic 的均值 / 標準差，衡量穩定度
  spread   : 前 20% 平均報酬 − 後 20% 平均報酬（扣掉市場整體漲跌）
  hit_rate : rank_ic > 0 的日數比例
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA


def evaluate(df: pd.DataFrame, quantile: float = 0.2, stride: int = 5) -> pd.DataFrame:
    """stride = 取樣間隔（交易日）。用來計算重疊造成的有效樣本折損。

    這是整個評估最容易被忽略、也最會騙人的地方：
      用 20 日期間、每 5 天取樣一次，相鄰兩個觀察有 15 天是重複的。
      名目上 47 個樣本，獨立資訊其實只有約 47/4 ≈ 12 個。
      不做這個調整，t 值會被高估約 sqrt(4)=2 倍 ——
      一個「t=3.6 高度顯著」的發現，調整後其實是「t=1.8 不顯著」。
    """
    out = []
    for (h, m), g in df.groupby(["horizon", "model"]):
        ics, spreads, tops = [], [], []
        for d, gd in g.groupby("as_of"):
            if len(gd) < 10 or gd["prob_up"].nunique() < 3:
                continue
            ic = gd["prob_up"].corr(gd["actual_return"], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
            k = max(int(len(gd) * quantile), 3)
            s = gd.sort_values("prob_up", ascending=False)
            mkt = gd["actual_return"].mean()
            top = s.head(k)["actual_return"].mean()
            bot = s.tail(k)["actual_return"].mean()
            spreads.append(top - bot)
            tops.append(top - mkt)          # 相對市場的超額
        if not ics:
            continue
        ics, spreads, tops = np.array(ics), np.array(spreads), np.array(tops)
        # spread 有自己的 t，與 RankIC 的 t 不同。原本表格只印 IC 的 t 卻放在
        # spread 欄位旁，讀者（含我自己寫 README 時）會誤以為那是 spread 的顯著性。
        t_sp = (float(spreads.mean() / (spreads.std(ddof=1) / np.sqrt(len(spreads))))
                if len(spreads) > 1 and spreads.std(ddof=1) > 0 else np.nan)
        overlap = max(h / stride, 1.0)          # 重疊倍數
        n_eff = len(ics) / overlap               # 有效獨立樣本數
        t_raw = (float(ics.mean() / (ics.std() / np.sqrt(len(ics))))
                 if ics.std() > 0 else np.nan)
        t_adj = t_raw / np.sqrt(overlap) if pd.notna(t_raw) else np.nan
        out.append({
            "horizon": h, "model": m, "days": len(ics),
            "n_eff": round(n_eff, 1), "t_adj": round(t_adj, 2) if pd.notna(t_adj) else np.nan,
            "spread_t": round(t_sp, 2) if pd.notna(t_sp) else np.nan,
            "spread_t_adj": (round(t_sp / np.sqrt(overlap), 2) if pd.notna(t_sp) else np.nan),
            "rank_ic": round(float(ics.mean()), 4),
            "ic_ir": round(float(ics.mean() / ics.std()), 3) if ics.std() > 0 else np.nan,
            "ic_hit_rate": round(float((ics > 0).mean()), 3),
            "spread": round(float(spreads.mean()), 5),
            "top_excess": round(float(tops.mean()), 5),
            "t_stat": round(t_raw, 2) if pd.notna(t_raw) else np.nan,
        })
    return pd.DataFrame(out).sort_values(["horizon", "rank_ic"], ascending=[True, False])


def report(res: pd.DataFrame) -> str:
    lines = []
    for h, g in res.groupby("horizon"):
        lines.append(f"\n{'='*86}")
        lines.append(f"  橫斷面選股能力   期間 {h} 個交易日   "
                     f"（問的是「挑得準不準」，不是「猜方向準不準」）")
        lines.append(f"{'='*86}")
        lines.append(f"  {'模型':<12}{'RankIC':>9}{'IC>0':>8}{'ICのt調整':>10}"
                     f"{'前20-後20':>11}{'spread調整t':>12}  判讀")
        lines.append("  " + "-" * 82)
        for _, r in g.iterrows():
            t, ta = r["t_stat"], r["t_adj"]
            if pd.isna(ta):
                note = "無變異（常數預測）"
            elif abs(ta) < 2.0:
                note = "調整後不顯著＝尚無證據"
            elif ta >= 2.0:
                note = "調整後仍顯著，值得前瞻驗證"
            else:
                note = "顯著為負（反向指標）"
            lines.append(f"  {r['model']:<12}{r['rank_ic']:>+9.4f}{r['ic_hit_rate']:>8.0%}"
                         f"{ta:>10.2f}{r['spread']:>+11.2%}"
                         f"{r['spread_t_adj']:>12.2f}  {note}")
        lines.append("  註1：always_up 給所有標的同一機率，橫斷面無排序能力，不列入比較。")
        lines.append("  註1b：RankIC 與 spread 各有自己的 t，兩欄不可互相代替；"
                     "判讀欄依 RankIC 的調整 t。")
        lines.append(f"  註2：{h} 日期間每 5 天取樣一次，相鄰樣本重疊 {max(h/5,1):.0f} 倍，")
        lines.append("       名目 t 值須除以 sqrt(重疊倍數) 才是可信的。判讀一律看「調整t」。")
    return "\n".join(lines)


if __name__ == "__main__":
    df = pd.read_parquet(DATA / "backtest/results.parquet")
    res = evaluate(df)
    print(report(res))
    res.to_parquet(DATA / "backtest/crosssec.parquet", index=False)
