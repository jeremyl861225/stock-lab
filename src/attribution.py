# -*- coding: utf-8 -*-
"""把一批判斷的成績拆成「市場判斷」與「選股判斷」兩項，分開計分。

為什麼非拆不可（2026-09-19 實測）：
  手寫 p20 的橫斷面標準差台股只有 0.027、美股 0.020，而錨點（整批的共同平移項）
  是 0.53／0.52。也就是說**每一筆判斷有 95% 以上的內容來自錨點**，
  只有剩下的幾個基點來自「這一檔與別檔有什麼不同」。
  混在一起算準確率與 Brier，量到的幾乎全是「錨點對不對」；
  而錨點是一天一個決定，選股是一天 50 個決定 ——
  兩者累積統計證據的速度差 50 倍，混在一起等於讓快的那個被慢的那個淹掉。

拆法（每個 as_of × market × horizon 各算一次）：
  市場判斷：錨點 − 0.5      對上  當日該市場實際上漲比率 − 0.5
  選股判斷：p − 該批的錨點   對上  個股報酬 − 該批的平均報酬（去均值）

  去均值之後，市場整體漲跌對選股成績完全沒有貢獻 ——
  這正是 crosssec.py 用 rank IC 的理由，但那裡只跑回測，
  而且不分市場（台美的交易日與漲跌互不相干，混在同一個橫斷面是錯的）。

錨點從哪裡來：
  1. 帳本（predictions.jsonl 的 anchor 欄）—— 2026-09-19 之後寫入的判斷都有。
  2. 判斷檔（judgments/*.json 的 anchor 欄）—— 之前的三天由此讀。
     錨點本來就寫在各 build 檔的 docstring 與 METHOD.md §4.3，
     把它補進判斷檔是把既有事實結構化，不是事後追加一個新決定。
  兩者都沒有時回傳 None，不臆測 —— 沒有錨點就是沒有辦法拆，該說出來。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SETTLEMENTS, ROOT


def anchors_from_judgments(jdir: Path | None = None) -> dict[tuple[str, str], float]:
    """回傳 {(as_of, market): anchor}，來源是判斷檔的 anchor 欄。"""
    jdir = jdir or (ROOT / "judgments")
    out: dict[tuple[str, str], float] = {}
    for f in sorted(jdir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        a = d.get("anchor")
        if a is None or "as_of" not in d:
            continue
        # 一年期的錨點是另一回事（rule_1y.ANCHOR），不參與 5／20 日的拆解
        if int(d.get("horizon", 20)) >= 250:
            continue
        out[(str(d["as_of"]), str(d.get("market", "TW")))] = float(a)
    return out


def _market_of(code: str) -> str:
    return "TW" if str(code).isdigit() else "US"


def decompose(df: pd.DataFrame, anchors: dict | None = None) -> pd.DataFrame:
    """df 需含 as_of, code, horizon, model, prob_up, actual_return。

    回傳每個 (horizon, model, market) 一列，市場判斷與選股判斷分開。
    """
    if df.empty:
        return pd.DataFrame()
    d = df.copy()
    if "market" not in d.columns:
        d["market"] = d["code"].map(_market_of)
    anchors = anchors or {}
    rows = []
    for (h, m, mk), g in d.groupby(["horizon", "model", "market"]):
        mkt_err, mkt_n = [], 0
        ics, spreads = [], []
        sel_hits, sel_n = 0, 0
        for as_of, gd in g.groupby("as_of"):
            if len(gd) < 5:
                continue
            realized_up = float((gd["actual_return"] > 0).mean())
            a = anchors.get((str(as_of), mk))
            if a is None and "anchor" in gd.columns:
                v = gd["anchor"].dropna()
                a = float(v.iloc[0]) if len(v) else None
            if a is not None:
                mkt_err.append(a - realized_up)
                mkt_n += 1
            # 選股：兩邊都去均值，市場整體漲跌因此完全不進來
            p = gd["prob_up"] - gd["prob_up"].mean()
            r = gd["actual_return"] - gd["actual_return"].mean()
            if p.std() > 0 and r.std() > 0:
                ic = float(pd.Series(p).corr(pd.Series(r), method="spearman"))
                if pd.notna(ic):
                    ics.append(ic)
                k = max(int(len(gd) * 0.2), 3)
                s = gd.assign(_p=p, _r=r).sort_values("_p", ascending=False)
                spreads.append(float(s.head(k)["_r"].mean() - s.tail(k)["_r"].mean()))
            sel_hits += int(((p > 0) == (r > 0)).sum())
            sel_n += len(gd)
        if not ics and not mkt_n:
            continue
        # 有效樣本：日頻預測在 h 日尺度上相鄰重疊 (h−1)/h，
        # 名目天數必須除以重疊倍數才是獨立資訊量。取樣間隔由資料本身推得。
        days = pd.to_datetime(pd.Series(g["as_of"].unique()),
                              format="%Y%m%d", errors="coerce").dropna()
        days = sorted(days)
        stride = 1.0
        if len(days) > 1:
            gaps = np.diff([d_.toordinal() for d_ in days])
            stride = max(float(np.median(gaps)), 1.0)
        overlap = max(h / stride, 1.0)
        ics_a = np.array(ics)
        t_adj = (float(ics_a.mean() / (ics_a.std(ddof=1) / math.sqrt(len(ics_a))))
                 / math.sqrt(overlap)
                 if len(ics_a) > 1 and ics_a.std(ddof=1) > 0 else np.nan)
        rows.append({
            "horizon": h, "model": m, "market": mk,
            "days": len(set(g["as_of"])), "n": int(len(g)),
            "n_eff_days": round(len(ics) / overlap, 1) if ics else 0.0,
            # 市場判斷：錨點減實際上漲比率。正值＝當時太樂觀。
            "market_bias": round(float(np.mean(mkt_err)), 4) if mkt_err else None,
            "market_mae": round(float(np.mean(np.abs(mkt_err))), 4) if mkt_err else None,
            "market_days": mkt_n,
            # 選股判斷：全部去均值後計算
            "sel_rank_ic": round(float(ics_a.mean()), 4) if ics else None,
            "sel_ic_t_adj": round(t_adj, 2) if pd.notna(t_adj) else None,
            "sel_spread": round(float(np.mean(spreads)), 5) if spreads else None,
            "sel_hit": round(sel_hits / sel_n, 4) if sel_n else None,
        })
    return pd.DataFrame(rows).sort_values(["horizon", "market", "model"])


def report(res: pd.DataFrame) -> str:
    if res.empty:
        return "（尚無已結算的預測，無法拆解）"
    L = ["", "=" * 92,
         "  判斷拆解：市場判斷 vs 選股判斷（兩者累積證據的速度差約 50 倍，不可混看）",
         "=" * 92,
         f"  {'期間':<5}{'市場':<5}{'模型':<12}{'天數':>5}{'錨點偏差':>10}"
         f"{'選股IC':>9}{'IC調整t':>9}{'前後20%':>10}{'有效天數':>9}  判讀"]
    L.append("  " + "-" * 88)
    for _, r in res.iterrows():
        mb = "—" if r["market_bias"] is None else f"{r['market_bias']:+.3f}"
        ic = "—" if r["sel_rank_ic"] is None else f"{r['sel_rank_ic']:+.4f}"
        t = "—" if r["sel_ic_t_adj"] is None else f"{r['sel_ic_t_adj']:+.2f}"
        sp = "—" if r["sel_spread"] is None else f"{r['sel_spread']*100:+.2f}%"
        if r["sel_ic_t_adj"] is None:
            note = "尚無選股證據"
        elif r["n_eff_days"] < 20:
            note = f"有效天數僅 {r['n_eff_days']:.0f}，任何結論都太早"
        elif abs(r["sel_ic_t_adj"]) < 2:
            note = "調整後不顯著＝尚無證據"
        else:
            note = "調整後顯著，值得追蹤"
        L.append(f"  {r['horizon']:<5}{r['market']:<5}{r['model']:<12}{r['days']:>5}"
                 f"{mb:>10}{ic:>9}{t:>9}{sp:>10}{r['n_eff_days']:>9.1f}  {note}")
    L.append("  " + "-" * 88)
    L.append("  錨點偏差 = 錨點 − 當日實際上漲比率，正值代表當時太樂觀（市場判斷的成績）。")
    L.append("  選股 IC／前後 20% 都已對當日該市場去均值，市場整體漲跌不貢獻分數。")
    L.append("  有效天數 = 名目天數 ÷ 重疊倍數（日頻預測、h 日期間，相鄰重疊 (h−1)/h）。")
    return "\n".join(L)


def load_settlements() -> pd.DataFrame:
    if not SETTLEMENTS.exists():
        return pd.DataFrame()
    rows = []
    for l in SETTLEMENTS.read_text(encoding="utf-8").splitlines():
        if l.strip():
            try:
                rows.append(json.loads(l))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def summary() -> dict:
    s = load_settlements()
    if s.empty:
        return {"status": "no_settlements"}
    s = s.sort_values("settled_at_utc").drop_duplicates(
        ["as_of", "horizon", "model", "code"], keep="last")
    s = s[s["horizon"] < 250]          # 一年期尚無結算，且錨點機制不同
    res = decompose(s, anchors_from_judgments())
    return {"status": "ok", "rows": int(len(s)),
            "by_group": res.to_dict("records") if not res.empty else []}


if __name__ == "__main__":
    s = load_settlements()
    if s.empty:
        print("尚無已結算的預測（data/settlements.jsonl 不存在或為空）。")
        print("第一批 5 日預測約在 2026-09-23 到期，20 日約在 2026-10-15。")
        a = anchors_from_judgments()
        print(f"\n目前判斷檔中已記錄錨點的批次：{len(a)} 批")
        for (d, mk), v in sorted(a.items()):
            print(f"  {d} {mk}  錨點 {v}")
        sys.exit(0)
    s = s.sort_values("settled_at_utc").drop_duplicates(
        ["as_of", "horizon", "model", "code"], keep="last")
    print(report(decompose(s[s["horizon"] < 250], anchors_from_judgments())))
