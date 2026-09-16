"""依期望值排序列出全部標的。

期望值 = P(up) × 上漲時幅度 + (1−P(up)) × 下跌時幅度
這才是可以拿來排序的數字：只看 P(up) 會漏掉幅度，
一檔 P(up)=0.6 但漲 1%／跌 3% 的標的，期望值是負的。

賠率比 = |上漲幅度 / 下跌幅度|，用來標註報酬與風險是否對稱。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, PREDICTIONS


def table(horizon: int = 20) -> pd.DataFrame:
    rows = [json.loads(l) for l in PREDICTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    p = pd.DataFrame([r for r in rows if r["model"] == "claude" and r["horizon"] == horizon])
    if p.empty:
        return p
    p = p.sort_values("created_at_utc").drop_duplicates("code", keep="last")
    b = pd.read_parquet(DATA / "briefing.parquet")
    m = p.merge(b[["code", "名稱", "產業", "PER", "dividend_yield", "rev_yoy",
                   "rsi_14", "ret_20", "foreign_5", "margin_chg_5", "dist_high_60"]],
                on="code", how="left")
    return m.sort_values("exp_ret", ascending=False).reset_index(drop=True)


def render(horizon: int = 20) -> str:
    t = table(horizon)
    if t.empty:
        return "（尚無判斷）"
    L = []
    L.append(f"{'='*136}")
    L.append(f"  期望值排序 · 未來 {horizon} 個交易日 · 基準日 {t['as_of'].iloc[0]} · "
             f"期望值 = P(漲)×漲幅 + P(跌)×跌幅")
    L.append(f"{'='*136}")
    L.append(f"  {'#':>2} {'代號':<5}{'名稱':<9}{'P(漲)':>6}{'漲幅':>7}{'跌幅':>7}"
             f"{'期望值':>8}{'賠率':>6} {'不對稱':<8}{'信心':<7}"
             f"{'RSI':>4}{'營收YoY':>9}{'20日':>7}{'外資5日':>8}  理由")
    L.append("  " + "─" * 132)
    for i, r in t.iterrows():
        yoy = "  n/a" if pd.isna(r["rev_yoy"]) else f"{r['rev_yoy']*100:+7.1f}%"
        tag = (r["asymmetry"] or "").replace("（上檔大）", "↑").replace("（下檔大）", "↓")
        L.append(f"  {i+1:>2} {r['code']:<5}{str(r['名稱'])[:8]:<9}"
                 f"{r['prob_up']:>6.2f}{r['up_magnitude']*100:>+6.1f}%{r['dn_magnitude']*100:>+6.1f}%"
                 f"{r['exp_ret']*100:>+7.2f}%{r['reward_risk']:>6.2f} {tag:<8}{r['conviction']:<7}"
                 f"{r['rsi_14']:>4.0f}{yoy:>9}{r['ret_20']*100:>+6.1f}%{r['foreign_5']*5:>+7.1f}x"
                 f"  {str(r['rationale']).split(': ',1)[-1][:46]}")
    pos = t[t["exp_ret"] > 0]; neg = t[t["exp_ret"] < 0]
    L.append("  " + "─" * 132)
    L.append(f"  正期望值 {len(pos)} 檔（均值 {pos['exp_ret'].mean()*100:+.2f}%）｜"
             f"負期望值 {len(neg)} 檔（均值 {neg['exp_ret'].mean()*100:+.2f}%）｜"
             f"全體均值 {t['exp_ret'].mean()*100:+.2f}%")
    asym = t[t["asymmetry"].astype(str).str.contains("偏")]
    L.append(f"  報酬風險不對稱者 {len(asym)} 檔："
             f"正偏 {len(t[t['asymmetry'].astype(str).str.contains('正偏')])} 檔、"
             f"負偏 {len(t[t['asymmetry'].astype(str).str.contains('負偏')])} 檔")
    return "\n".join(L)


if __name__ == "__main__":
    for h in (20, 5):
        print(render(h)); print()
