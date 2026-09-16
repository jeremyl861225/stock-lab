# -*- coding: utf-8 -*-
"""判斷品質自動查核 —— 把四輪人工審核的工作變成每天自動跑的閘門。

四輪獨立審核共抓到 30+ 個錯誤，沒有一個是「模型不夠聰明」，全部是紀律問題：
沒查證就寫、用錯的變數、時間錯置、單邊採證。這支程式把其中可機械化的部分
攔在判斷進入系統之前。

檢查項目（每一項都對應 LESSONS.md 裡真實發生過的錯誤）：
  1. 數值宣稱     why 裡的每個數字要能在 briefing 對得上          （E1：緯穎的數字串到台塑化）
  2. 極值宣稱     「最高／最低／唯一／全場」用排序驗              （B1：16 次誤述）
  3. 自我矛盾     同一份文件內互相衝突的極值主張                  （B1：UNH 與 PG 都宣稱「全場最弱」）
  4. 論述一致性   機率與宣稱的主導維度，相關方向對不對            （F1：部位沒實作論述）
  5. 單邊採證     與判斷方向相反的極端值是否被略過                （D1：集中在信心最高的檔位）
  6. 新聞覆蓋     漲跌幅前十與 high/medium 判斷是否有新聞可查     （C1：歸因錯誤的根源）
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RAW

# 欄位 → (比對用的 regex, 是否為百分比)
# 每個 pattern 都用 negative lookbehind 排除會前綴它的詞 ——
# 否則「融資20日+19.5%」會被當成 ret_20、「5日內」會被當成 ret_5。
FIELDS = {
    "margin_chg_5":  (r"融資\s*5\s*日", True),
    "margin_chg_20": (r"融資\s*20\s*日", True),
    "ret_5":  (r"(?<!融資)(?<!融資 )\b5\s*日(?!均)", True),
    "ret_20": (r"(?<!融資)(?<!融資 )\b20\s*日(?!均)", True),
    "ret_60": (r"(?<!融資)\b60\s*日(?!高|均)", True),
    "dist_high_60": (r"距\s*(?:60\s*日)?高(?:點)?", True),
    "rsi_14": (r"RSI", False),
    "rev_yoy": (r"營收(?:年增|年減|僅|\+)", True),
    "PER": (r"PER", False),
    "PBR": (r"PBR", False),
    "dividend_yield": (r"殖利率", False),
    "vol_20": (r"日波動", True),
}
# 只檢查「無限定範圍」的極值宣稱。
# 「金融股最強」「傳產最優」「大型科技PER最低」是有範圍的，用全體排序驗會誤判。
GLOBAL_MARK = ["全場", "所有", "唯一", "全 universe", "全庫"]


def _window(why: str, start: int, limit: int = 18) -> str:
    """從 start 取窗口，但遇到「下一個欄位名」就截斷。

    否則「日波動5.9%全場最高」裡的『最高』會被前面的「5日」認領 ——
    實測這正是最後兩個誤報的成因。
    """
    end = min(start + limit, len(why))
    for _pat, _ in FIELDS.values():
        m = re.compile(_pat).search(why, start)
        if m and start < m.start() < end:
            end = m.start()
    return why[start:end]


def _load(as_of: str) -> tuple[pd.DataFrame, list[dict]]:
    b = pd.read_parquet(DATA / "briefing.parquet")
    js = []
    for f in sorted(Path("judgments").glob(f"{as_of}*.json")):
        js.append(json.loads(f.read_text(encoding="utf-8")))
    # 同一 (市場, horizon) 取 mtime 最新
    best: dict[tuple, dict] = {}
    for f in sorted(Path("judgments").glob(f"{as_of}*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (d.get("market", "TW"), d["horizon"])
        if key not in best or f.stat().st_mtime > best[key]["_mt"]:
            d["_mt"] = f.stat().st_mtime
            best[key] = d
    return b, [d for k, d in best.items() if k[1] == 20]


def check_numbers(b: pd.DataFrame, judgments: list[dict], tol: float = 0.08) -> list[str]:
    """why 裡的每個數字要能在該檔的 briefing 那一列找到對應。"""
    out = []
    bi = b.set_index("code")
    for d in judgments:
        for j in d["judgments"]:
            code, why = j["code"], j.get("thesis", "")
            if code not in bi.index:
                continue
            row = bi.loc[code]
            for field, (pat, is_pct) in FIELDS.items():
                if field not in row or pd.isna(row[field]):
                    continue
                actual = float(row[field]) * (100 if is_pct else 1)
                rx = re.compile(pat + r"\s*(?:約|達|為|是)?\s*([+\-−]?\d+\.?\d*)\s*%?")
                # 括號內常是「（全場最優是緯穎PER6.9）」這種引用別檔的數字，不算本檔宣稱
                masked = re.sub(r"（[^）]*）", lambda m: "　" * len(m.group(0)), why)
                for m in rx.finditer(masked):
                    raw = m.group(1).replace("−", "-")
                    try:
                        claimed = float(raw)
                    except ValueError:
                        continue
                    # 「營收年減16%」的 16 沒帶負號，方向由「減」字決定
                    if claimed > 0 and re.search(r"年減|下滑|衰退", m.group(0)):
                        claimed = -claimed
                    if abs(claimed - actual) > max(abs(actual) * tol, 0.15):
                        out.append(f"[數值] {code} 「{m.group(0).strip()}」→ 實際 {actual:.2f}")
    return out


def check_superlatives(b: pd.DataFrame, judgments: list[dict]) -> list[str]:
    """極值宣稱逐一用排序驗，但只驗「全場／所有／唯一」這類無限定範圍的。"""
    out = []
    WORDS = {"最高": False, "最強": False, "最大": False,
             "最低": True, "最弱": True, "最小": True, "最深": True}
    for d in judgments:
        mk = d.get("market", "TW")
        sub = b[b["market"] == mk]
        codes = set(sub["code"])
        for j in d["judgments"]:
            why, code = j.get("thesis", ""), j["code"]
            if code not in codes or not any(g in why for g in GLOBAL_MARK):
                continue
            for field, (pat, _) in FIELDS.items():
                if field not in sub or sub[field].isna().all():
                    continue
                for word, asc in WORDS.items():
                    # 欄位名與極值詞必須靠近（30 字內），否則是不同句子的兩件事
                    for m in re.finditer(pat, why):
                        # 窗口要窄：30 字會把「日波動5.9%全場最高」配到前面的 ret_5。
                        seg = _window(why, m.end())
                        if word not in seg or not any(g in seg for g in GLOBAL_MARK):
                            continue
                        # 「全場第2高」「全場次強」「全場最低十分位」都不是在宣稱第一
                        if re.search(r"第\s*\d|次[高低強弱]|分位|之列|之一", seg):
                            continue
                        s2 = sub[["code", field]].dropna()
                        order = s2.sort_values(field, ascending=asc)["code"].tolist()
                        if code in order:
                            rank = order.index(code) + 1
                            if rank > 1:
                                out.append(f"[極值] {code} 宣稱「{field} {word}（全場）」"
                                           f"實為第 {rank} 名，首位 {order[0]}")
                        break
    return sorted(set(out))


def check_contradictions(b: pd.DataFrame, judgments: list[dict]) -> list[str]:
    """同一份文件內互相衝突的「全場最X」主張。"""
    out = []
    for d in judgments:
        mk = d.get("market", "TW")
        claims: dict[str, list[str]] = {}
        for j in d["judgments"]:
            why = j.get("thesis", "")
            if not any(g in why for g in GLOBAL_MARK):
                continue
            for field, (pat, _) in FIELDS.items():
                for m in re.finditer(pat, why):
                    seg = _window(why, m.end())
                    if not any(g in seg for g in GLOBAL_MARK):
                        continue
                    if re.search(r"第\s*\d|次[高低強弱]|分位|之列|之一", seg):
                        continue
                    for word in ("最高", "最低", "最強", "最弱", "最小", "最大"):
                        if word in seg:
                            claims.setdefault(f"{mk} {field} {word}", []).append(j["code"])
        for k, codes in claims.items():
            if len(set(codes)) > 1:
                out.append(f"[矛盾] {k}：{sorted(set(codes))} 同時宣稱")
    return out


def check_thesis_alignment(b: pd.DataFrame, judgments: list[dict]) -> list[str]:
    """機率與宣稱的主導維度，相關方向對不對。"""
    out = []
    for d in judgments:
        mk = d.get("market", "TW")
        macro = d.get("market_context", "")
        df = pd.DataFrame([{"code": j["code"], "p": j["prob_up"]} for j in d["judgments"]])
        sub = b[b["market"] == mk][["code", "rsi_14", "rev_yoy", "ret_20"]]
        m = df.merge(sub, on="code").dropna()
        if len(m) < 10:
            continue
        r_rsi = m["p"].corr(m["rsi_14"])
        r_rev = m["p"].corr(m["rev_yoy"])
        out.append(f"[論述] {mk} p vs RSI {r_rsi:+.3f}｜p vs 營收年增 {r_rev:+.3f}")
        if "基本面" in macro and r_rev < 0.15:
            out.append(f"[論述] ！{mk} 主張看重基本面，但 p 與營收相關僅 {r_rev:+.3f}"
                       f"（部位可能沒有實作論述）")
    return out


def check_one_sided(b: pd.DataFrame, judgments: list[dict]) -> list[str]:
    """與判斷方向相反、且落在前後 12% 的極端值，是否在理由裡被提及。"""
    out = []
    for d in judgments:
        mk = d.get("market", "TW")
        sub = b[b["market"] == mk]
        for j in d["judgments"]:
            code, why, p = j["code"], j.get("thesis", ""), j["prob_up"]
            if abs(p - 0.5) < 0.03 or j.get("conviction") == "low":
                continue
            row = sub[sub["code"] == code]
            if row.empty:
                continue
            row = row.iloc[0]
            bearish = p < 0.5
            for field, (aliases, _) in FIELDS.items():
                if field not in sub or pd.isna(row.get(field)):
                    continue
                s = sub[field].dropna()
                if len(s) < 10:
                    continue
                pct = (s < row[field]).mean()
                # 看空卻有很正面的極端值（或反之），且理由完全沒提到這個欄位
                contra = (bearish and pct > 0.88) or (not bearish and pct < 0.12)
                # 殖利率對成長股天生低，不構成「與看多相反的證據」
                positive_fields = {"rev_yoy"}
                if field in positive_fields and contra and not any(a in why for a in aliases):
                    out.append(f"[單邊] {code}（p={p:.2f}）未提 {field} 位於 pct{pct*100:.0f}，"
                               f"方向與判斷相反")
    return out


def check_news_coverage(b: pd.DataFrame, judgments: list[dict], as_of: str) -> list[str]:
    """漲跌幅前十與 high/medium 判斷，是否有新聞檔可查。"""
    out = []
    nd = RAW / "news" / as_of
    for d in judgments:
        mk = d.get("market", "TW")
        sub = b[b["market"] == mk].dropna(subset=["ret_20"])
        extremes = set(sub.nlargest(5, "ret_20")["code"]) | set(sub.nsmallest(5, "ret_20")["code"])
        for j in d["judgments"]:
            code = j["code"]
            need = code in extremes or j.get("conviction") in ("high", "medium")
            if not need:
                continue
            f = nd / f"{code}.json"
            n = len(json.loads(f.read_text(encoding="utf-8")).get("payload", [])) if f.exists() else 0
            if n == 0:
                out.append(f"[新聞] {code}（{j.get('conviction')}）無新聞可查證，歸因風險高")
    return out


def run(as_of: str) -> dict:
    b, js = _load(as_of)
    if not js:
        return {"error": f"找不到 judgments/{as_of}*.json"}
    groups = {
        "數值宣稱": check_numbers(b, js),
        "極值宣稱": check_superlatives(b, js),
        "自我矛盾": check_contradictions(b, js),
        "論述一致性": check_thesis_alignment(b, js),
        "單邊採證": check_one_sided(b, js),
        "新聞覆蓋": check_news_coverage(b, js, as_of),
    }
    print(f"═══ 判斷品質查核（{as_of}）═══")
    total = 0
    for name, items in groups.items():
        warn = [x for x in items if not x.startswith("[論述] " + x[5:6]) or "！" in x or "[論述]" not in x]
        real = [x for x in items if "[論述]" not in x or "！" in x]
        total += len(real)
        print(f"\n── {name}：{len(real) if real else '通過'}"
              f"{' 項待確認' if real else ''}")
        for x in items[:12]:
            print(f"   {x}")
        if len(items) > 12:
            print(f"   …另有 {len(items)-12} 項")
    print(f"\n═══ 合計 {total} 項待確認 ═══")
    print("這些不一定全是錯誤，但每一項都要人工確認過才能進系統。")
    return {"as_of": as_of, "issues": total, "groups": {k: len(v) for k, v in groups.items()}}


if __name__ == "__main__":
    import pandas as _pd
    a = sys.argv[1] if len(sys.argv) > 1 else _pd.read_parquet(
        DATA / "features/panel.parquet", columns=["date"])["date"].max().strftime("%Y%m%d")
    run(a)
