# -*- coding: utf-8 -*-
"""美股判斷 2026-09-16。幅度與台股同一套：base = 0.8 × vol_20 × √h。

與台股最大的差別：**美股沒有籌碼面**。
台股每日公告三大法人買賣超與融資餘額，美股沒有對應的每日資料，
所以四面向只剩基本面、技術面、新聞面三面。信心普遍低於台股同類判斷。
"""
import json, math, sys
from pathlib import Path
import pandas as pd

MACRO = (
 "美股當前主題是『AI 放緩疑慮』：費半近期重挫、科技巨擘公開呼籲放緩 AI 發展，"
 "Broadcom CEO 甚至需出面回應並重申 2,300 億美元 AI 晶片營收目標。"
 "這使 AI 硬體（NVDA／AVGO／TSM）與平台股（GOOGL／META／MSFT／AMZN）出現分化："
 "前者估值與情緒同時承壓，後者相對絕緣。"
 "注意：美股無每日法人買賣超與融資餘額公告，本判斷只有基本面、技術面、新聞面三個面向，"
 "因此信心普遍低於台股同類判斷。TSM 與台股 2330 是同一家公司，"
 "ADR 相對台股仍有約一成溢價，溢價收斂時 ADR 會較弱。")

# code, p_up(20日), skew, conviction, why
J = [
 ("AVGO",0.55, 0.12,"medium","營收年增85.5%但RSI33.8全場最低、距高-20.2%；CEO公開重申2300億AI營收目標，跌深來自情緒而非基本面"),
 ("GOOGL",0.55, 0.10,"medium","PER17.3為大型科技股最低、營收年增24.2%、距高-8.5%；AI放緩疑慮對平台股衝擊小於硬體股"),
 ("AMZN",0.54, 0.08,"low","RSI40.9、20日-4.5%、距高-12.7%；PER20相對營收年增19.6%不算貴"),
 ("NVDA",0.53, 0.06,"low","營收年增105.9%但PER僅27.3，成長與估值相稱；惟為AI放緩疑慮的核心標的，雙向風險大"),
 ("TSM",0.52, 0.02,"low","營收年增36%、距高-12.4%；與台股2330同一公司，ADR溢價約一成，溢價收斂會壓抑ADR表現"),
 ("MSFT",0.51, 0.02,"low","RSI53.7、20日+2.6%、PER27.5、營收+17.7%；無明確催化劑"),
 ("QQQ",0.51, 0.02,"low","那斯達克100代理，RSI48.2、20日-1.0%；接近市場中性"),
 ("VOO",0.51, 0.02,"low","S&P500代理，分散度高於QQQ、日波動僅0.5%；接近市場中性"),
 ("BRK-B",0.50, 0.00,"low","PER13.0全場最低、日波動0.6%最小；RSI66.7偏高但防禦屬性明確，無方向"),
 ("BTCO",0.50, 0.00,"low","比特幣ETF，20日已漲17.2%、日波動3.1%；其驅動邏輯獨立於股市，我沒有可據以判斷的依據，明確表示無觀點"),
 ("AAPL",0.48,-0.04,"low","PER38.2偏高但營收年增僅16.4%為大型科技股最弱；5日已漲5.5%、RSI62.3"),
 ("META",0.47,-0.08,"low","20日+24.4%全場最強、RSI72.8最高；營收+28%與PER25.5尚稱相稱，但多篇看多預測顯示情緒已充分反映"),
 ("TSLA",0.47,-0.06,"low","PER338.7為全場極端值、日波動3.2%；20日+7.6%後估值風險大於成長"),
]


def build(horizon: int, vol: dict) -> dict:
    out = []
    for code, p20, skew, conf, why in J:
        p = 0.5 + (p20 - 0.5) * math.sqrt(horizon / 20)
        base = 0.8 * vol.get(code, 0.02) * math.sqrt(horizon)
        up, dn = base * (1 + skew), -base * (1 - skew)
        ev = p * up + (1 - p) * dn
        ratio = abs(up / dn) if dn else float("inf")
        tag = "正偏（上檔大）" if ratio > 1.25 else ("負偏（下檔大）" if ratio < 0.8 else "對稱")
        out.append({
            "code": code, "stance": "bullish" if ev >= 0 else "bearish",
            "conviction": conf, "prob_up": round(p, 3), "exp_ret": round(ev, 5),
            "ret_q10": round(dn * 1.6, 5), "ret_q90": round(up * 1.6, 5),
            "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
            "reward_risk": round(ratio, 2), "asymmetry": tag,
            "thesis": why, "facts": ["四面向資料見 data/briefing.parquet（美股無籌碼面）"],
            "inference": why,
            "falsifier": "若 AI 放緩疑慮擴大為實際的 capex 下修（雲端業者財測轉向），"
                         "AI 硬體的『跌深錯殺』論點失效；反之若疑慮消退，超買平台股的回檔論點失效。",
        })
    return {"as_of": "20260916", "horizon": horizon, "analyst": "claude-opus-5",
            "market": "US", "version": "v1", "market_context": MACRO, "judgments": out}


if __name__ == "__main__":
    b = pd.read_parquet("data/briefing.parquet")
    b = b[b["market"] == "US"]
    vol = dict(zip(b["code"], b["vol_20"].fillna(0.02)))
    miss = [c for c, *_ in J if c not in vol]
    assert not miss, f"代號不在美股 universe：{miss}"
    for h in (20, 5):
        d = build(h, vol)
        Path(f"judgments/20260916_us_h{h}.json").write_text(
            json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        ev = [x["exp_ret"] for x in d["judgments"]]
        print(f"US h={h}: {len(d['judgments'])} 檔，期望值 {min(ev)*100:+.2f}% ~ {max(ev)*100:+.2f}%")
