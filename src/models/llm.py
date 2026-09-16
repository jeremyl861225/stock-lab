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

VERSION = "1.0.0"
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
    lines = []
    for r in rows:
        n = r["news"][:5]
        news_txt = "\n".join(f"      - {x}" for x in n) if n else "      （無近期財經新聞）"
        lines.append(
            f"""  {r['code']} {r['name']}
      收盤 {r['close']:.1f}｜近5日 {r['ret_5']:+.2%}｜近20日 {r['ret_20']:+.2%}｜近60日 {r['ret_60']:+.2%}
      20日波動 {r['vol_20']:.2%}｜距20日均線 {r['ma_gap_20']:+.2%}｜RSI {r['rsi_14']:.0f}｜距60日高點 {r['dist_high_60']:+.2%}
      外資近5日買賣超(佔均量) {r['foreign_5']:+.2f}｜投信 {r['trust_5']:+.2f}
      近期新聞：
{news_txt}""")
    mkt_txt = "\n".join(f"  - {x}" for x in mkt[:8]) or "  （無）"
    return f"""你是量化研究員。請為以下台股標的預測「{as_of} 收盤後、未來 {horizon} 個交易日」的漲跌。

嚴格要求：
1. 輸出「校準過的」機率。意思是：你給 0.70 的那一批標的，實際上必須有大約 70% 上漲。
   歷史基本率（同期間上漲比例）約為 {base_rate:.1%} —— 沒有明確理由時，請貼近這個數字。
2. 不要為了顯得有觀點而給極端值。多數個股在 {horizon} 個交易日尺度上接近隨機。
3. 只使用下列資訊，不要引用你記憶中 {as_of} 之後發生的任何事。

大盤背景：
{mkt_txt}

標的：
{chr(10).join(lines)}

輸出 JSON 陣列，每檔一個物件，不要有其他文字：
[{{"code":"2330","prob_up":0.55,"rationale":"20字以內的理由"}}]"""

def predict(feats: pd.DataFrame, horizon: int, as_of, panel, uni: dict,
            base_rate: float = 0.52) -> pd.DataFrame:
    if _provider() is None:
        return pd.DataFrame()
    as_of_str = pd.Timestamp(as_of).strftime("%Y%m%d")
    names = {c["code"]: c["name"] for c in uni["constituents"]}
    mkt = [x["title"] for x in news.market_news(as_of_str)]

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
                out.append({"code": str(o["code"]), "prob_up": p,
                            "direction": 1 if p >= 0.5 else -1,
                            "rationale": f"llm: {str(o.get('rationale',''))[:60]}"})
        except Exception as e:  # noqa: BLE001
            print(f"  llm batch {i} failed: {e}")
    return pd.DataFrame(out)

ALL = {"llm": predict}
