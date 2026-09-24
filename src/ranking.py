"""依期望值排序列出全部標的。

**5／20 日**：期望值 = P(up) × 上漲時幅度 + (1−P(up)) × 下跌時幅度。
這才是可以拿來排序的數字：只看 P(up) 會漏掉幅度，
一檔 P(up)=0.6 但漲 1%／跌 3% 的標的，期望值是負的。

**250 日（一年）**：`exp_ret` 是同一個對數常態的**中位數**，不是上面那條兩點式
（見 `models/price_1y.py`）。用平均數排序會退化成純波動度排序 ——
對數常態的平均數含 exp(σ²/2)，σ=0.8 時是中位數的 1.38 倍。
兩個期別的欄位名相同但語意不同，改動這支時要分開想。

賠率比 = |上漲幅度 / 下跌幅度|，用來標註報酬與風險是否對稱。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, PREDICTIONS


def table(horizon: int = 20, market: str | None = None) -> pd.DataFrame:
    rows = [json.loads(l) for l in PREDICTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    p = pd.DataFrame([r for r in rows if r["model"] == "claude" and r["horizon"] == horizon])
    if p.empty:
        return p
    p = p.sort_values("created_at_utc").drop_duplicates("code", keep="last")
    b = pd.read_parquet(DATA / "briefing.parquet")
    cols = ["code", "名稱", "產業", "PER", "dividend_yield", "rev_yoy", "rsi_14",
            "ret_20", "foreign_5", "margin_chg_5", "dist_high_60", "close", "vol_20",
            "market", "權重", "vol_60"]
    m = p.merge(b[[c for c in cols if c in b.columns]], on="code", how="left",
                suffixes=("", "_b"))
    # 只取各市場自己最新的 as_of。
    #
    # 要先 merge 再篩 —— 帳本的列本身沒有 market 欄位，市場是從 briefing 帶進來的。
    # 若在 merge 前用全域 max 篩，台股推進到新交易日之後，還停在前一日的美股
    # 判斷會被整組濾掉，美股分頁直接空掉（實測 2026-09-17 台股推進後美股歸零）。
    # 兩個市場的收盤時間差 12 小時以上，as_of 本來就會不同步，這是常態不是例外。
    #
    # 仍然要篩的理由不變：某天沒做判斷而 briefing 照常重建時，不篩會靜默把
    # 昨天的幅度配上今天的收盤，算出錯的獲利點與停損點，且不會有任何訊號。
    if not m.empty and "market" in m.columns:
        m = m[m["as_of"] == m.groupby("market")["as_of"].transform("max")]
    if market:
        m = m[m["market"] == market]
    m = _attach_judged_close(m, b)
    return m.sort_values("exp_ret", ascending=False).reset_index(drop=True)


def _attach_judged_close(m: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """補上 close_judged（判斷日的收盤）與 briefing_as_of。

    上面那道 as_of 篩選比的是帳本 as_of 在各市場內的最大值，從來沒比 briefing 的 as_of。
    某個市場當天沒做新判斷、briefing 卻照常重建時（2026-09-21～23 美股連續如此），
    卡片會拿 9/22 的收盤配 9/21 的幅度：21 檔價位最多偏 5%（MU），JPM 的價位漂移
    等於整段預估漲幅 —— 獲利點幾乎只是隔天的收盤價。
    修法不是把 close 換掉（那會讓收盤價與同列的 RSI、K 線最後一根對不上），
    而是另給一個「判斷日收盤」：獲利點／停損點／中位價以它為基準，
    收盤價那格照顯示最新值並標日期。panel 不在版控內（CI 沒有）時退回 briefing 的 close。
    """
    if m.empty:
        return m
    m = m.copy()
    b_as_of = (b.groupby("market")["as_of"].max() if "market" in b.columns and "as_of" in b.columns
               else pd.Series(dtype="datetime64[ns]"))
    m["briefing_as_of"] = m["market"].map(
        lambda mk: pd.Timestamp(b_as_of[mk]).strftime("%Y%m%d") if mk in b_as_of.index else None)
    m["close_judged"] = m["close"]
    pnl_p = DATA / "features/panel.parquet"
    stale = m["briefing_as_of"].notna() & (m["briefing_as_of"] != m["as_of"])
    if stale.any() and pnl_p.exists():
        pnl = pd.read_parquet(pnl_p, columns=["date", "code", "close"])
        pnl["ds"] = pnl["date"].dt.strftime("%Y%m%d")
        px = pnl.set_index(["code", "ds"])["close"]
        for i in m.index[stale]:
            key = (str(m.at[i, "code"]), str(m.at[i, "as_of"]))
            if key in px.index:
                m.at[i, "close_judged"] = float(px[key])
    return m


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
