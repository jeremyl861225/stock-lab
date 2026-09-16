"""LLM 模型：讀新聞＋量化摘要後做判斷。

三個刻意的設計決定：
1. Provider 無關（Anthropic / Gemini），沒有 API key 就安靜跳過，
   絕不讓整條管線因為單一模型失效而中斷。
2. Prompt 明確要求「校準過的機率」並給出基本率錨點。LLM 天生過度自信，
   不壓這一手的話 Brier score 會很難看。
3. LLM 無法回測（它已經知道歷史結果），所以它的成績只能從上線日往後算，
   而且必須跟 baseline 並排看 —— 這正是本系統存在的理由。
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect import news
from briefing import fundamentals


def _pct(v):
    return "n/a" if v is None or pd.isna(v) else f"{v*100:+.1f}%"

VERSION = "1.2.0"
BATCH = 10

def _provider():
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return None

def _call(prompt: str) -> str:
    import requests
    p = _provider()
    if p == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                     "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.getenv("LLM_MODEL", "claude-sonnet-5"),
                  "max_tokens": 2000,
                  "messages": [{"role": "user", "content": prompt}]}, timeout=120)
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    if p == "gemini":
        m = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"],
                     "content-type": "application/json"},
            json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=120)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    raise RuntimeError("no LLM provider configured")

def _prompt(rows: list[dict], horizon: int, as_of: str, base_rate: float,
            mkt: list[str]) -> str:
    """分析框架與 2026-09-16 人工判斷所用的完全一致，確保自動化後可重現。"""
    lines = []
    for r in rows:
        n = r["news"][:5]
        news_txt = "\n".join(f"      - {x}" for x in n) if n else "      （無近期財經新聞）"
        lines.append(
            f"""  {r['code']} {r['name']}（{r.get('industry','')}）收盤 {r['close']:.1f}
      基本面：PER {r.get('per','n/a')}｜PBR {r.get('pbr','n/a')}｜殖利率 {r.get('yield','n/a')}%
              月營收 {r.get('rev_month','n/a')} 年增 {r.get('rev_yoy','n/a')}｜近3月年增 {r.get('rev_yoy3m','n/a')}｜月增 {r.get('rev_mom','n/a')}
      籌碼面：外資5日 {r['foreign_5']:+.2f}／20日 {r.get('foreign_20',0):+.2f} 倍日均量｜投信5日 {r['trust_5']:+.2f}
              融資餘額5日 {r.get('margin_chg_5',0):+.1%}｜20日 {r.get('margin_chg_20',0):+.1%}
      技術面：RSI {r['rsi_14']:.0f}｜5日 {r['ret_5']:+.2%}｜20日 {r['ret_20']:+.2%}｜60日 {r['ret_60']:+.2%}
              距60日高點 {r['dist_high_60']:+.2%}｜距20日均線 {r['ma_gap_20']:+.2%}｜日波動 {r['vol_20']:.2%}
      新聞：
{news_txt}""")
    mkt_txt = "\n".join(f"  - {x}" for x in mkt[:10]) or "  （無）"
    return f"""你是量化研究員。針對以下台股標的，預測 {as_of} 收盤後、未來 {horizon} 個交易日的報酬分布。

分析必須同時涵蓋四個面向，並在理由中指出是哪個面向主導你的判斷：
  基本面：營收年增與月增動能、PER／PBR 是否與成長相稱、殖利率
  籌碼面：外資與投信買賣超方向是否一致、融資餘額變化（融資急增常是散戶追高的反向訊號）
  技術面：RSI 是否極端、距高點位置、動能是否已過度延伸
  新聞面：是否有新資訊（新資訊才會推動股價；已公告並被消化的利多不算）

三個必須遵守的紀律：
1. 輸出「校準過的」機率。你給 0.60 的那批標的，實際上必須約有 60% 上漲。
   歷史基本率約 {base_rate:.1%}，沒有明確理由時請貼近它。多數個股在 {horizon} 個交易日尺度上接近隨機。
2. 分開給「上漲時的幅度」與「下跌時的幅度」，不要只給單一預期值。
   兩者不同時即為報酬風險不對稱，這是判斷中最有價值的部分。
   幅度量級請參照各股日波動 × sqrt({horizon})，勿明顯偏離。
3. 只使用下列資訊。不得引用你記憶中 {as_of} 之後發生的任何事。

大盤背景：
{mkt_txt}

標的：
{chr(10).join(lines)}

輸出 JSON 陣列，每檔一物件，不要有其他文字：
[{{"code":"2330","prob_up":0.55,"up_magnitude":0.08,"dn_magnitude":-0.07,
   "conviction":"low","rationale":"40字內，須點明主導面向"}}]"""


def predict(feats: pd.DataFrame, horizon: int, as_of, panel, uni: dict,
            base_rate: float = 0.52) -> pd.DataFrame:
    if _provider() is None:
        return pd.DataFrame()
    as_of_str = pd.Timestamp(as_of).strftime("%Y%m%d")
    names = {c["code"]: c["name"] for c in uni["constituents"]}
    inds = {c["code"]: c.get("industry", "") for c in uni["constituents"]}
    mkt = [x["title"] for x in news.market_news(as_of_str)]
    fd = fundamentals(as_of_str)
    fund = {r["code"]: {**r, "industry": inds.get(r["code"], "")}
            for r in fd.to_dict("records")} if not fd.empty else {}

    rows = []
    for _, r in feats.iterrows():
        code = r["code"]
        rows.append({
            "code": code, "name": names.get(code, code), "close": r["close"],
            "ret_5": r["ret_5"] or 0, "ret_20": r["ret_20"] or 0,
            "ret_60": r["ret_60"] or 0, "vol_20": r["vol_20"] or 0,
            "ma_gap_20": r["ma_gap_20"] or 0, "rsi_14": r["rsi_14"] or 50,
            "dist_high_60": r["dist_high_60"] or 0,
            "foreign_5": r["foreign_5"] or 0, "trust_5": r["trust_5"] or 0,
            "foreign_20": r.get("foreign_20") or 0,
            "margin_chg_5": r.get("margin_chg_5") or 0,
            "margin_chg_20": r.get("margin_chg_20") or 0,
            "ret_60": r.get("ret_60") or 0, "ma_gap_20": r.get("ma_gap_20") or 0,
            "vol_20": r.get("vol_20") or 0.02,
            "industry": fund.get(code, {}).get("industry", ""),
            "per": fund.get(code, {}).get("PER"), "pbr": fund.get(code, {}).get("PBR"),
            "yield": fund.get(code, {}).get("dividend_yield"),
            "rev_month": fund.get(code, {}).get("rev_month"),
            "rev_yoy": _pct(fund.get(code, {}).get("rev_yoy")),
            "rev_yoy3m": _pct(fund.get(code, {}).get("rev_yoy3m")),
            "rev_mom": _pct(fund.get(code, {}).get("rev_mom")),
            "news": [x["title"] for x in news.stock_news(code, names.get(code, code), as_of_str)],
        })

    out = []
    for i in range(0, len(rows), BATCH):
        chunk = rows[i:i + BATCH]
        try:
            txt = _call(_prompt(chunk, horizon, as_of_str, base_rate, mkt))
            m = re.search(r"\[.*\]", txt, re.S)
            for o in json.loads(m.group(0) if m else txt):
                p = float(np.clip(o.get("prob_up", base_rate), 0.01, 0.99))
                up = abs(float(o.get("up_magnitude", 0.05)))
                dn = -abs(float(o.get("dn_magnitude", -0.05)))
                ev = p * up + (1 - p) * dn
                out.append({"code": str(o["code"]), "prob_up": p, "exp_ret": ev,
                            "ret_q10": dn * 1.35, "ret_q90": up * 1.35,
                            "up_magnitude": up, "dn_magnitude": dn,
                            "reward_risk": round(abs(up / dn), 2) if dn else None,
                            "direction": 1 if ev >= 0 else -1,
                            "rationale": f"llm[{o.get('conviction','?')}]: "
                                         f"{str(o.get('rationale',''))[:70]}"})
        except Exception as e:  # noqa: BLE001
            print(f"  llm batch {i} failed: {e}")
    return pd.DataFrame(out)

ALL = {"llm": predict}
