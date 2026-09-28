# -*- coding: utf-8 -*-
"""美股判斷 · 基準日 2026-09-25（美股資料日即 9/25；台股 9/25 中秋休市，as_of 仍為 9/24 且已有判斷）。

上一份美股判斷的資料日是 9/23，所以本份涵蓋 9/24、9/25 兩個新交易日。
美股 9/25 的基準線與統計模型已由 GitHub Actions 排程（台北 9/26 03:14）寫進帳本，本機只補判斷。

**本日主軸：溫和反彈，沒有單一主題。**
  9/25 briefing 53 檔等權 +0.30%、34/53 收紅；VOO +0.54%、QQQ +0.46%。
  9/24 為 panel 68 檔等權 −0.12%、31/68 收紅。

**具名事件**：ORCL 發出資料中心不可抗力通知（Project Jupiter），9/24 −3.47%、5 日 −7.12% 全場最弱；
  QCOM 與 Apple 授權續約（9/25 +3.97% 全場最強）；COST 9/24 盤後公布財報、9/25 +2.93%；
  MU 財報 9/30 落在 5 日視窗內。

**p 的構成**：p = 錨點 + 0.025×(2×營收成長百分位−1) + 0.01×(2×便宜度百分位−1)，
  另有三檔手動覆寫。美股無籌碼面、營收是季頻，信心一律 low。不對 20 日漲幅扣分（反轉 IC 為負）。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.quantiles import quantiles

AS_OF = "20260925"
DATA_DATE = "2026-09-25"
ANCHOR = 0.52   # METHOD §4.3；US 首三批 h5 已結算，有效天數 0.6，不調錨

ETF = {"VOO", "QQQ", "BTCO"}
BANKS = {"JPM", "BAC", "C", "WFC", "GS", "MS", "AXP"}

MACRO = (
    "資料日 9/25（美東）：briefing 53 檔等權 +0.30%、34/53 收紅；VOO +0.54%、QQQ +0.46%。"
    "前一日 9/24 為 panel 68 檔等權 −0.12%、31/68 收紅。溫和反彈，沒有單一主題。\n\n"
    "**具名事件**：ORCL 9/24 發出資料中心不可抗力通知（Project Jupiter，Morningstar 估可能遞延約 250 億美元營收），"
    "當日 −3.47%，5 日 −7.12% 為全場最低；QCOM 9/25 +3.97% 為全場最高，Bernstein 同日針對 Apple 授權續約重申評等；"
    "COST 9/24 盤後公布財報，DA Davidson 上調目標價，9/25 +2.93%；MSFT 9/25 +3.66%（Oppenheimer 參訪後調目標價）；"
    "PANW 9/25 −3.89% 為全場最低，新聞檔無成因。MU 財報 9/30 落在 5 日視窗內。"
    "新聞檔中刊於 9/26–9/27 的報導（MS 的 UBS 合併傳聞、CRM 再遭降評）晚於資料日，不作為本判斷依據。\n\n"
    "**RSI 最低的是 RTX 29.34（全場最低）**，其次 BAC 31.34、MS 31.87、HD 31.91。"
    "RTX 有具名的正面基本面（上調財測、未交貨訂單 2,890 億美元），而跌勢成因未查明；依 LESSONS 2026-09-18，"
    "超賣本身不加分。六家銀行 10/13–14 公布財報，落在 20 日視窗內，是一個賭注押六次。\n\n"
    "**一年期的『財報已更新』是假警報**：TSM／NVDA／AVGO／META／AAPL／MU 的季報快照 0 筆新季度、0 筆數值變動，"
    "雜湊變動來自本益比小數位（與 9/21–9/24 同一成因）。MU 的新季度要到 9/30 財報後才會進來。\n\n"
    "**p 的主導維度是基本面**：季營收年增排序（±0.025）加本益比便宜度（±0.01）；ETF 三檔、MU、QCOM 棄權，"
    "ORCL 因具名利空覆寫到錨點之下。**不做反轉**：不對 20 日漲幅扣分，這段樣本反轉的 IC 為負。"
    "美股無籌碼面、營收季頻，信心一律 low。錨點維持 0.52。"
)

NOTE = {
    "MU": (0.52, "財報 9/30 落在 5 日視窗內，是二元事件；營收年增 +345.7% 是循環高點的分母，"
           "不能與其他成長股同尺度排序。故棄權，p＝錨點。"),
    "ORCL": (0.51, "9/24 發出資料中心不可抗力通知（Project Jupiter），Morningstar 估可能遞延約 250 億美元營收；"
             "季營收年增 +29.6% 是落後指標，而這則事件打的正是前瞻營收（LESSONS C1）。"
             "規則值看多，但不讓落後指標蓋過具名利空，覆寫到錨點之下。"),
    "QCOM": (0.52, "季營收年增 −4.0% 為全場最低（規則因此看空），但 9/25 與 Apple 授權續約的報導"
             "正好移除了一項前瞻營收風險。兩者相抵、方向無法事前判斷，棄權，p＝錨點。"),
    "COST": (None, "9/24 盤後已公布財報（DA Davidson 上調目標價），事件已過；下一次財報 12/10 在視窗外。"),
    "RTX": (None, "RSI 29.34 為全場最低；有具名正面（上調財測、未交貨訂單 2,890 億美元）但跌勢成因未查明，"
            "超賣不加分。財報 10/20 在 20 日視窗內。"),
    "PANW": (None, "9/25 −3.89% 為全場最低、新聞檔無成因；本益比 914 是小分母，便宜度排序墊底屬預期。"),
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
        facts = (f"{code}：9/25 {pct(r['ret_1'])}、5 日 {pct(r['ret_5'])}、20 日 {pct(r['ret_20'])}、"
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
                inf += "10 月中六家銀行同週公布財報，是 20 日期間內的主事件；同產業同時公布是一個賭注押六次，本檔不另加減。"
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
            "facts": [thesis, f"美股資料日 {DATA_DATE}；briefing 53 檔等權 +0.30%、34/53 檔收紅"],
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
