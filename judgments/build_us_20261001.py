# -*- coding: utf-8 -*-
"""美股判斷 · 基準日 2026-10-01（美股資料日 10/1；台股 as_of 為 10/2）。

**本日主軸：持平。** briefing 53 檔等權 −0.05%、中位 +0.02%、27/53 收紅。

**MU 財報已落地**（9/30 盤後），棄權理由由「二元事件未揭曉」改為「循環高點分母」。

**p 的構成**：同 9/30 —— p = 錨點 + 0.025×(2×營收成長百分位−1) + 0.01×(2×便宜度百分位−1)，
  另有三檔手動覆寫（MU、ORCL、QCOM）。美股無籌碼面、營收是季頻，信心一律 low。不對 20 日漲幅扣分。
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.quantiles import quantiles

AS_OF = "20261001"
DATA_DATE = "2026-10-01"
ANCHOR = 0.52   # METHOD §4.3；US h5 已結算有效天數約 2，不調錨

ETF = {"VOO", "QQQ", "BTCO"}
BANKS = {"JPM", "BAC", "C", "WFC", "GS", "MS", "AXP"}

MACRO = (
    "資料日 10/1（美東）：briefing 53 檔等權 −0.05%、中位 +0.02%、27/53 收紅；接近持平，漲跌家數各半。\n\n"
    "**下跌端**：AMGN −3.38%（全場最低；新聞多為 9/28–29 的目標價上調與股利，當日跌幅查不到具名事件）、NFLX −2.49%、JNJ −2.30%、AVGO −2.15%、C −1.92%。"
    "**上漲端**：CRM +3.10%（全場最高）、MU +3.03%（9/30 盤後財報：營收與財測優於預期、多家上調目標價）、IBM +2.59%、CAT +1.92%、PLTR +1.60%。\n\n"
    "**5 日最弱**：META −6.6%（全場最低）、TSLA −6.3%、QCOM −6.3%。**20 日**：AMD +34.7% 為全場最高、NFLX −18.0% 為全場最低。"
    "**RSI 最低**：BAC 22.1（全場最低）、RTX 24.4、MS 24.6；依 LESSONS 2026-09-18，超賣本身不加分。\n\n"
    "**事件視窗**：5 日視窗內無財報。20 日視窗內有 30 檔財報：C、WFC、GS、JPM、UNH、JNJ 10/13（8 個營業日後）、MS、BAC 10/14，TSM 10/15，"
    "其後 GE、RTX、NFLX、TXN、TSLA、IBM、PM、PG、AXP、VZ、KO、V，最後一批 META、MSFT、GOOGL、CAT、MA、LLY、AMZN、MRK、AAPL 落在 19–20 個營業日後的視窗邊緣。\n\n"
    "**一年期**：美股 10/1 的一年期已由排程（GitHub Actions）滾動，本機為同日不重滾；新聞閘本日無命中。\n\n"
    "**p 的主導維度是基本面**：季營收年增排序（±0.025）加本益比便宜度（±0.01）；ETF 三檔、MU、QCOM 棄權，"
    "ORCL 因具名利空覆寫到錨點之下。**不做反轉**：不對 20 日漲幅扣分。"
    "美股無籌碼面、營收季頻，信心一律 low。錨點維持 0.52（US 已結算有效天數約 2，不足以調錨）。"
)

NOTE = {
    "MU": (0.52, "9/30 盤後財報已落地，本日 +3.03% 已反映；新一季營收年增 +379.3% 為全場最高、本益比降到 14.7。"
           "但那是記憶體循環高點的分母：成長排序會把它排第一、便宜度也會加分，兩項都在獎勵週期位置而非可持續的成長。"
           "與一年期手寫論點（記憶體本益比最低時最貴）一致，棄權，p＝錨點。"),
    "ORCL": (0.51, "9/24 發出資料中心不可抗力通知（Project Jupiter），Morningstar 估可能遞延約 250 億美元營收；"
             "本日新聞仍只有看多評論與遞延估算，沒有解除事件的消息。"
             "季營收年增 +29.6% 是落後指標，而這則事件打的正是前瞻營收（LESSONS C1）。維持覆寫到錨點之下（沿用 9/28）。"),
    "QCOM": (0.52, "季營收年增 −4.0% 為全場最低（規則因此看空），但 9/24 Apple 授權續約移除了一項前瞻營收風險；"
             "5 日 −6.3% 為全場第 3 低，新聞歸因 AI 與 Apple 題材降溫，兩股力量方向相反。棄權，p＝錨點（沿用 9/28）。"),
    "BAC": (None, "RSI 22.1 為全場最低；超賣不加分。"),
    "AMGN": (None, "本日 −3.38% 為全場最低；新聞為目標價上調與調升股利，查不到當日具名利空（查不到不等於查無，LESSONS 2026-09-22），不另加減。"),
    "NFLX": (None, "20 日 −18.0% 為全場最低；新聞為 Deutsche Bank 升評。不做反轉，財報 10/20 在 20 日視窗內。由規則計分。"),
    "PANW": (None, "本益比 1007.1 為全場最高（小分母），便宜度排序墊底屬預期；BTIG 上調目標價至 425 美元。"),
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
        facts = (f"{code}：10/1 {pct(r['ret_1'])}、5 日 {pct(r['ret_5'])}、20 日 {pct(r['ret_20'])}、"
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
            "facts": [thesis, f"美股資料日 {DATA_DATE}；briefing 53 檔等權 −0.05%、27/53 檔收紅"],
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
