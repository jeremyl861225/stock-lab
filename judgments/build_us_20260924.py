# -*- coding: utf-8 -*-
"""美股判斷 · 檔名基準日 2026-09-24（美股資料實際最新交易日為 2026-09-23）。

prepare 印出的 as_of 是台股的 20260924；美股在 JST 15:44 時 9/24 尚未開盤，
briefing 的 US 列 as_of 為 2026-09-23。predict／settle 逐檔以各自市場的交易日對帳
（LESSONS 2026-09-17），檔名沿用 prepare 指定的日期。

**本日主軸：金融股集體重挫而查不到具名原因。**
  9/23 等權 −0.67%、27/68 收紅（panel）。WFC −5.4%、MS −3.8%、AXP −3.7%、GS −2.4%、C −2.3%。
  新聞檔裡沒有能解釋這個族群同步下跌的標題（WFC 的新聞都是它對別家的評等）。
  不編造原因：金融股依規則計分，並在理由裡明寫「成因未查明」。

**p 的構成**：p = 錨點 + 0.025×(2×營收成長百分位−1) + 0.01×(2×便宜度百分位−1)。
  美股無籌碼面、營收是季頻，信心一律 low。不對 20 日漲幅扣分（反轉 IC 為負）。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.quantiles import quantiles

AS_OF = "20260924"
DATA_DATE = "2026-09-23"
ANCHOR = 0.52   # METHOD §4.3；US 已結算 21 筆、有效天數 0.2，不調錨

ETF = {"VOO", "QQQ", "BTCO"}
BANKS = {"JPM", "BAC", "C", "WFC", "GS", "MS", "AXP"}

MACRO = (
    "資料日 9/23（美東）：本系統 panel 等權 −0.67%、27/68 檔收紅；VOO −0.7%、QQQ −0.8%。\n\n"
    "**金融股集體重挫、成因未查明**：WFC −5.4%、MS −3.8%、AXP −3.7%、GS −2.4%、C −2.3%。"
    "本系統新聞檔中找不到解釋族群同步下跌的標題（WFC 的新聞多是它對他股的評等），"
    "故不編造原因。六家銀行 10/13–14 公布財報，落在 20 日視窗內。\n\n"
    "**個股具名事件**：AMGN +3.3%（Sjögren's 第三期試驗正面、Jefferies 上調目標價）、"
    "PANW +5.8%（新聞檔無相關標題，成因未查明）、MU 財報 9/30 落在 5 日視窗內"
    "（Citi 上調目標價、DRAM 報價）。\n\n"
    "**一年期的「財報已更新」是假警報**：美股季報快照 0 筆新季度（seen 皆早於 9/22），"
    "唯一變動的雜湊輸入是本益比小數位，與 9/21–9/23 同一成因。\n\n"
    "**p 的主導維度是基本面**：季營收年增排序（±0.025）加本益比便宜度（±0.01）；"
    "ETF 三檔與 MU 棄權。**不做反轉**：不對 20 日漲幅扣分，這段樣本反轉的 IC 為負。"
    "美股無籌碼面、營收季頻，信心一律 low。錨點維持 0.52。"
)

NOTE = {
    "MU": (0.52, "財報 9/30 落在 5 日視窗內，是二元事件；營收年增數百個百分點是循環高點的分母，"
           "不能與其他成長股同尺度排序。故棄權，p＝錨點。"),
    "AMGN": (None, "新聞具名：Sjögren's 第三期試驗正面、Jefferies 上調目標價，9/23 +3.3%。"
             "營收成長偏低，規則值不高；新聞是既有管線的正面讀出，不另加分。"),
    "PANW": (None, "9/23 +5.8% 但新聞檔無相關標題，成因未查明；本益比 914 是小分母，便宜度排序墊底屬預期。"),
    "COST": (None, "事件欄為空，財報日未知；不據推測的日期調整。"),
}


def pct(x):
    return f"{x*100:+.1f}%"


def make(b: pd.DataFrame):
    b = b.set_index("code")
    st = b[~b.index.isin(ETF)]
    g = st["rev_yoy"].rank(pct=True)
    v = (-st["PER"]).rank(pct=True)
    n_rev = int(st["rev_yoy"].notna().sum())
    J = []
    for code, r in b.iterrows():
        name = str(r["名稱"])[:28]
        facts = (f"{code}：9/23 {pct(r['ret_1'])}、5 日 {pct(r['ret_5'])}、20 日 {pct(r['ret_20'])}、"
                 f"距 60 日高點 {pct(r['dist_high_60'])}、RSI {r['rsi_14']:.1f}、日波動 {r['vol_20']*100:.1f}%。")
        if code not in ETF and pd.notna(r["rev_yoy"]):
            rk = int(st["rev_yoy"].rank(ascending=False)[code])
            facts += f"季營收年增 {pct(r['rev_yoy'])}（個股有值 {n_rev} 檔中第 {rk} 高）、"
        if pd.notna(r["PER"]):
            facts += f"本益比 {r['PER']:.1f}。"
        if bool(r.get("evt_in_20")):
            facts += f"20 日視窗內有{r['evt_type']}（{r['evt_date']}，{int(r['evt_days'])} 個營業日後）。"
        thr = max(0.04, round(0.8 * r["vol_20"] * math.sqrt(20), 2))
        if code in ETF:
            p = ANCHOR
            inf = "指數或商品 ETF 沒有個股基本面可排序，棄權，p＝錨點。"
            fal = f"棄權檔。若 {code} 20 日內絕對報酬逾 {thr*100:.0f}%（約 1σ），代表市場層級有可事前判斷的方向。"
        else:
            gp = g.get(code) if pd.notna(g.get(code, np.nan)) else 0.5
            vp = v.get(code) if pd.notna(v.get(code, np.nan)) else 0.5
            rule_p = round(ANCHOR + 0.025 * (2 * gp - 1) + 0.01 * (2 * vp - 1), 3)
            ov, extra = NOTE.get(code, (None, ""))
            p = rule_p if ov is None else ov
            inf = (f"規則起算：營收成長百分位 {gp:.2f}、便宜度百分位 {vp:.2f} → 規則值 {rule_p:.3f}"
                   + (f"，手動覆寫為 {p:.3f}（理由見下）。" if ov is not None else "。"))
            if ov is None:
                inf += ("看多的理由是營收成長在排序前段而估值未墊高到抵銷。" if p > ANCHOR else
                        "看空的理由是營收成長落後或估值偏貴，而非股價延伸。" if p < ANCHOR else
                        "規則落在錨點。")
            if code in BANKS:
                inf += "9/23 金融族群同步下跌、成因未查明，本檔不因此加減；10 月中財報是 20 日期間內的主事件。"
            if extra:
                inf += extra
            if p > ANCHOR:
                fal = (f"若 {code} 20 日內下跌逾 {thr*100:.0f}%（約 1σ），代表本檔的看多理由無效；"
                       f"5 日內下跌逾 {thr*50:.0f}% 為早期警訊。")
            elif p < ANCHOR:
                fal = f"若 {code} 20 日內上漲逾 {thr*100:.0f}%（約 1σ），代表本檔的看空理由不構成壓力。"
            else:
                fal = f"棄權檔。若 {code} 5 日內單邊變動逾 {thr*50:.0f}%，代表事件方向其實可以事前判斷。"
        skew = round((p - ANCHOR) * 2, 3)
        J.append((code, p, skew, "low", facts, inf, fal))
    return J


def build(J, horizon: int, vol: dict) -> dict:
    out = []
    for code, p20, skew, conf, thesis, inf, fal in J:
        p = 0.5 + (p20 - 0.5) * math.sqrt(horizon / 20)
        base = 0.85 * vol.get(code, 0.02) * math.sqrt(horizon)
        up, dn = base * (1 + skew), -base * (1 - skew)
        ev = p * up + (1 - p) * dn
        sig = vol.get(code, 0.02) * math.sqrt(horizon)
        q10, q90 = quantiles(ev, sig)
        ev = round(ev, 5)
        ratio = abs(up / dn) if dn else float("inf")
        tag = "正偏（上檔大）" if ratio > 1.25 else ("負偏（下檔大）" if ratio < 0.8 else "對稱")
        out.append({
            "code": code, "stance": "bullish" if ev >= 0 else "bearish",
            "conviction": conf, "prob_up": round(p, 3), "exp_ret": ev,
            "ret_q10": round(q10, 5), "ret_q90": round(q90, 5),
            "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
            "reward_risk": round(ratio, 2), "asymmetry": tag,
            "thesis": thesis, "inference": inf,
            "facts": [thesis, f"美股資料日 {DATA_DATE}；panel 等權 −0.67%、27/68 檔收紅"],
            "falsifier": fal,
        })
    return {"as_of": AS_OF, "horizon": horizon, "analyst": "claude-opus-5",
            "market": "US", "version": "v1", "anchor": ANCHOR,
            "market_context": MACRO, "judgments": out}


if __name__ == "__main__":
    b = pd.read_parquet("data/briefing.parquet")
    b = b[b["market"] == "US"]
    assert b["as_of"].astype(str).str[:10].eq(DATA_DATE).all(), "美股 briefing 日期不符"
    vol = dict(zip(b["code"], b["vol_20"].fillna(0.02)))
    J = make(b)
    codes = [c for c, *_ in J]
    assert len(codes) == len(set(codes)) == len(b)
    ps = pd.Series([x[1] for x in J])
    print(f"  p 標準差 {ps.std():.4f}、範圍 {ps.min():.3f}–{ps.max():.3f}、"
          f"高於錨點 {(ps > ANCHOR).sum()}、等於 {(ps == ANCHOR).sum()}、低於 {(ps < ANCHOR).sum()}")
    for h, suffix in ((20, ""), (5, "_h5")):
        d = build(J, h, vol)
        p = Path("judgments") / f"{AS_OF}_us{suffix or '_h20'}.json"
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        ev = [x["exp_ret"] for x in d["judgments"]]
        print(f"  h={h:<3} {len(d['judgments'])} 檔 → {p.name}  "
              f"期望值 {min(ev)*100:+.2f}% ~ {max(ev)*100:+.2f}%")
