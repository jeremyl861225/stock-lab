"""Claude 的判斷 —— 帶推理鏈、棄權機制與否證條件。

這是本系統中唯一無法自動化、也無法回測的模型，理由與設計都寫在這裡。

為什麼不能回測：我知道歷史。叫我判斷 2025 年 3 月的台積電，我對之後發生
什麼的知識會污染判斷，即使努力不用。所以成績只能從上線日往後累積。

為什麼值得加進來（數學模型沒有的三件事）：
  1. 棄權：沒有論點時不出手。stat_gbdt 對 50 檔都會給一個數字，
     但其中大部分是雜訊。只在有論點時出手，能提高每次判斷的資訊密度，
     也縮短證明「我到底有沒有 edge」所需的時間。
  2. 推理鏈：facts / inference 分開記錄，錯誤因此可被歸因 ——
     是事實錯了（可換資料源）、推論錯了（我的問題）、
     還是推論對但市場不理會（市場效率，該放棄這個角度）。
     gbdt 給你 0.73，你永遠不知道為什麼，也無從改進。
  3. 否證條件：事先寫下「什麼證據會證明我錯」。這逼我把模糊直覺
     變成可檢驗的命題，也讓事後檢討無法自圓其說。

輸出與其他模型一致（prob_up / exp_ret / ret_q10 / ret_q90），
才能被同一套計分程式公平地打分。
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA

VERSION = "1.3.0"
REASONING = DATA / "reasoning.jsonl"
REQUIRED = {"code", "stance", "prob_up", "exp_ret", "ret_q10", "ret_q90",
            "conviction", "thesis", "facts", "inference", "falsifier"}


def load(path: Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    for j in d["judgments"]:
        miss = REQUIRED - set(j)
        if miss:
            raise ValueError(f"{j.get('code')} 缺欄位：{miss}")
        if j["stance"] == "abstain":
            continue
        if not (0 < j["prob_up"] < 1):
            raise ValueError(f"{j['code']} prob_up 超出範圍")
        if not (j["ret_q10"] <= j["exp_ret"] <= j["ret_q90"]):
            raise ValueError(f"{j['code']} 期望報酬未落在 q10~q90 之間")
        if (j["exp_ret"] >= 0) != (j["stance"] == "bullish"):
            raise ValueError(f"{j['code']} stance 與 exp_ret 方向不一致")
        if not j["facts"] or not j["falsifier"]:
            raise ValueError(f"{j['code']} 缺少事實依據或否證條件")
    return d


def to_frame(d: dict) -> pd.DataFrame:
    """只回傳實際出手的標的；棄權者不進入計分。"""
    rows = [j for j in d["judgments"] if j["stance"] != "abstain"]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "code": str(j["code"]), "prob_up": float(j["prob_up"]),
        "exp_ret": float(j["exp_ret"]), "ret_q10": float(j["ret_q10"]),
        "ret_q90": float(j["ret_q90"]),
        "direction": 1 if j["exp_ret"] >= 0 else -1,
        "rationale": f"claude[{j['conviction']}]: {j['thesis'][:120]}",
    } for j in rows])


def save_reasoning(d: dict, pid_map: dict[str, str]) -> int:
    """推理鏈另存，用 pid 與預測記錄關聯 —— 事後歸因的唯一依據。"""
    REASONING.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with REASONING.open("a", encoding="utf-8") as f:
        for j in d["judgments"]:
            rec = dict(j)
            rec["as_of"] = d["as_of"]
            rec["horizon"] = d["horizon"]
            rec["pid"] = pid_map.get(str(j["code"]), "")
            rec["market_context"] = d.get("market_context", "")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n
