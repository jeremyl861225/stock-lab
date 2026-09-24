# -*- coding: utf-8 -*-
"""每週迭代：讓系統自己量出哪裡錯、把證據擺好，人只做分類與決定。

用法：
    ./.venv/bin/python src/iterate.py                      # 產出本週迭代包，append data/iterations.jsonl
    ./.venv/bin/python src/iterate.py --gate <results.parquet> [--name X]   # 候選模型晉升關卡

這支程式**不改任何模型、不改帳本、不改判斷**。它做的是把 WEEKLY.md 第一到第三節
原本要人工翻檔案的事變成一個指令，並把每次的結果 append 進 data/iterations.jsonl ——
讓「上週說要改什麼、改了沒、改了有沒有變準」有一條可回溯的軌跡。

迭代包六段（設計理由見 ITERATE.md）：
  A 成績單         分市場、逐日群集 t、重疊折減；有效天數不足就標「太早」
  B 判斷拆解       市場判斷（錨點 − 實際）與選股判斷（去均值 rank IC）分開
  C 錯誤歸因包     claude 錯最大的 N 檔 ＋ 當時的論點與否證條件 ＋ 否證條件的機械判定
  D 區間           覆蓋率、上／下方越界比率（越界方向與錨點偏差同號）
  E 查核器         verify_judgment 對最近一批的警告數（查核器要跟著 LESSONS 長）
  F 外洩守門       回測結果對照天花板；候選模型要先過這關

「改法有沒有用」的判準只有一個：**前瞻結算**的分市場、折減後 t。
回測只用來排除（外洩、明顯劣於猜漲），不用來宣稱有效。
"""
from __future__ import annotations
import datetime as dt, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, ROOT, SETTLEMENTS
import score as score_mod
import attribution as attr_mod
import accuracy as acc_mod

ITER_LOG = DATA / "iterations.jsonl"
OUT_DIR = ROOT / "research" / "iterations"
WORST_N = 12
MIN_EFF_DAYS_GATE = 20        # 候選模型每個 (市場, 期別) 至少要有的有效天數
T_GATE = 2.0                  # 折減後 t 的門檻


# ── A／B／D：成績與拆解 ───────────────────────────────────────────────

def scorecard() -> list[dict]:
    s = score_mod.summary()
    rows = []
    for h, mkts in s.get("by_horizon_market", {}).items():
        for mk, blk in mkts.items():
            for m in blk["models"]:
                if m["version_split"]:
                    continue
                rows.append({"horizon": int(h), "market": mk, "model": m["model"],
                             "n": m["n"], "n_abstain": m["n_abstain"], "n_days": m["n_days"],
                             "n_eff_days": m["n_eff_days"], "accuracy": m["accuracy"],
                             "accuracy_by_prob": m["accuracy_by_prob"],
                             "brier": m["brier"], "brier_skill": m["brier_skill"],
                             "edge": m["edge_vs_always_up"], "edge_t_adj": m["edge_t_adj"],
                             "p": m["p_value_vs_base"], "coverage_80": m["coverage_80"],
                             "interval_score": m["interval_score"]})
    return rows


def intervals() -> list[dict]:
    """區間：覆蓋率之外，上方／下方越界分開列 —— 首批 claude 12 檔越界全在上方，
    那是市場判斷偏空的另一個讀數，比覆蓋率本身更有資訊。"""
    df = score_mod._load()
    if df.empty:
        return []
    df = score_mod.prepare(df)
    out = []
    for (h, mk, m), g in df.groupby(["horizon", "market", "model"]):
        g = g.dropna(subset=["in_interval"])
        if g.empty:
            continue
        out.append({"horizon": int(h), "market": mk, "model": m, "n": int(len(g)),
                    "coverage_80": round(float(g["in_interval"].mean()), 4)})
    return out


# ── C：錯誤歸因包 ─────────────────────────────────────────────────────

_FALS = re.compile(r"(\d+)\s*(?:個交易)?日內.*?(漲|反彈|跌|回落|回檔).*?逾?\s*(\d+(?:\.\d+)?)\s*%")


def _reasoning_latest() -> dict[str, dict]:
    p = DATA / "reasoning.jsonl"
    if not p.exists():
        return {}
    out = {}
    for l in p.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        if r.get("pid"):
            out[r["pid"]] = r          # 後寫的覆蓋前面的 → 最新修訂
    return out


def _panel_moves(pnl: pd.DataFrame, code: str, as_of: str, n: int) -> tuple[float, float] | None:
    g = pnl[pnl["code"] == code].sort_values("date")
    ds = g["date"].dt.strftime("%Y%m%d").tolist()
    if as_of not in ds:
        return None
    i = ds.index(as_of)
    win = g.iloc[i + 1:i + 1 + n]
    if win.empty:
        return None
    p0 = float(g.iloc[i]["close"])
    return float(win["high"].max() / p0 - 1), float(win["low"].min() / p0 - 1)


def falsifier_check(text: str, pnl: pd.DataFrame, code: str, as_of: str) -> str:
    """把「N 日內漲逾 X%」這類否證條件機械判定。判不了的回 'unparseable'。"""
    m = _FALS.search(text or "")
    if not m:
        return "unparseable"
    n, word, pct = int(m.group(1)), m.group(2), float(m.group(3)) / 100
    mv = _panel_moves(pnl, code, as_of, n)
    if mv is None:
        return "pending"
    hi, lo = mv
    hit = (hi >= pct) if word in ("漲", "反彈") else (lo <= -pct)
    return "hit" if hit else "not_hit"


def attribution_packet(model: str = "claude") -> dict:
    df = score_mod._load()
    if df.empty:
        return {"worst": [], "falsifier": {}}
    df = score_mod.prepare(df)
    d = df[(df["model"] == model) & (df["horizon"] < 250)].copy()
    d = d.sort_values("settled_at_utc").drop_duplicates(["as_of", "horizon", "code"], keep="last")
    reason = _reasoning_latest()
    pnl_p = DATA / "features/panel.parquet"
    pnl = pd.read_parquet(pnl_p, columns=["date", "code", "close", "high", "low"]) if pnl_p.exists() else None
    stats = {"hit": 0, "not_hit": 0, "pending": 0, "unparseable": 0, "shared_text": 0}
    texts = {}
    for _, r in d.iterrows():
        rs = reason.get(r["pid"], {})
        f = rs.get("falsifier", "")
        texts.setdefault((r["as_of"], r["horizon"]), set()).add(f)
        if pnl is not None:
            stats[falsifier_check(f, pnl, str(r["code"]), str(r["as_of"]))] += 1
    # 整批共用同一句否證的批次數（CJ-03：9/16 起每個 5／20 日檔皆然）
    stats["shared_text"] = sum(1 for v in texts.values() if len(v) == 1)
    stats["batches"] = len(texts)
    wrong = d[(d["correct_eff"] == 0)].copy()
    wrong["abs_ret"] = wrong["actual_return"].abs()
    worst = []
    for _, r in wrong.sort_values("abs_ret", ascending=False).head(WORST_N).iterrows():
        rs = reason.get(r["pid"], {})
        worst.append({"as_of": r["as_of"], "horizon": int(r["horizon"]), "code": r["code"],
                      "prob_up": r["prob_up"], "exp_ret": r.get("exp_ret"),
                      "actual_return": r["actual_return"],
                      "thesis": (rs.get("thesis") or "")[:160],
                      "falsifier": (rs.get("falsifier") or "")[:120],
                      "classify_as": ""})       # 人填：事實錯／推論錯／市場不理會
    return {"worst": worst, "falsifier": stats}


# ── E：查核器覆蓋 ────────────────────────────────────────────────────

def verifier_summary() -> dict:
    try:
        import verify_judgment as V
        js = sorted((ROOT / "judgments").glob("2*.json"))
        if not js:
            return {}
        as_of = max(json.loads(f.read_text(encoding="utf-8")).get("as_of", "") for f in js)
        import contextlib, io, os
        cwd = os.getcwd(); os.chdir(ROOT)            # verify_judgment 用相對路徑找 judgments/
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = V.run(as_of)
        finally:
            os.chdir(cwd)
        return {"as_of": as_of, "issues": res.get("issues"), "by_category": res.get("groups", {})}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)[:200]}


# ── F：外洩守門與候選模型關卡 ─────────────────────────────────────────

def leak_flags() -> list[dict]:
    try:
        import leak_canary as L
        rp, cp = DATA / "backtest/results.parquet", L.OUT
        if not (rp.exists() and cp.exists()):
            return []
        return L.guard(pd.read_parquet(rp), json.loads(cp.read_text(encoding="utf-8"))["oracles"])
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)[:200]}]


def gate(cand: pd.DataFrame, name: str = "candidate") -> dict:
    """候選模型晉升關卡。輸入與 backtest.results 同 schema 的 walk-forward 結果（PIT 列）。

    通過的條件（每個 (市場, 期別) 各自判）：
      1. 有效天數 ≥ MIN_EFF_DAYS_GATE；
      2. 沒有冒煙（leak_canary.guard）；
      3. 對 always_up 的逐日配對 Brier 差，折減後 t ≤ −T_GATE（Brier 顯著較低）
         或 配對準確率差 t ≥ T_GATE；
      4. rank IC 折減後 t ≥ T_GATE（要有選股資訊，不是只把基本率校準得更好）。
    通過只代表「可以以新名字影子上線」，不代表有效 —— 有效要看前瞻結算。
    """
    import leak_canary as L
    base_p = DATA / "backtest/results.parquet"
    base = pd.read_parquet(base_p) if base_p.exists() else pd.DataFrame()
    base = base[base["model"] == "always_up"] if not base.empty else base
    orc_p = L.OUT
    orc = json.loads(orc_p.read_text(encoding="utf-8"))["oracles"] if orc_p.exists() else []
    cand = cand[cand["pit"]] if "pit" in cand.columns else cand
    out = {"name": name, "cells": [], "pass": False}
    flags = L.guard(cand.assign(model=name), orc) if orc else []
    for (mk, h), g in cand.groupby(["market", "horizon"]):
        cell = {"market": mk, "horizon": int(h), "n": int(len(g))}
        j = g.merge(base[base["horizon"] == h][["as_of", "code", "correct", "brier"]],
                    on=["as_of", "code"], suffixes=("", "_b")) if not base.empty else pd.DataFrame()
        if j.empty:
            cell.update({"reason": "沒有可配對的 always_up 基準"}); out["cells"].append(cell); continue
        ov = score_mod._overlap(j, int(h))
        d_acc = j.assign(d=j["correct"] - j["correct_b"]).groupby("as_of")["d"].mean()
        d_br = j.assign(d=j["brier"] - j["brier_b"]).groupby("as_of")["d"].mean()
        t_acc, n_eff = score_mod.cluster_t(d_acc, ov)
        t_br, _ = score_mod.cluster_t(d_br, ov)
        ics = [gd["prob_up"].corr(gd["actual_return"], method="spearman")
               for _, gd in g.groupby("as_of") if len(gd) >= 10 and gd["prob_up"].nunique() >= 3]
        ics = pd.Series([x for x in ics if pd.notna(x)])
        t_ic, _ = score_mod.cluster_t(ics, ov)
        smoke = any(f.get("market") == mk and f.get("horizon") == h for f in flags)
        ok = (n_eff >= MIN_EFF_DAYS_GATE and not smoke
              and ((t_br is not None and t_br <= -T_GATE) or (t_acc is not None and t_acc >= T_GATE))
              and (t_ic is not None and t_ic >= T_GATE))
        cell.update({"n_eff_days": n_eff, "paired_acc": round(float(d_acc.mean()), 4),
                     "t_acc": t_acc, "paired_brier": round(float(d_br.mean()), 5), "t_brier": t_br,
                     "rank_ic": round(float(ics.mean()), 4) if len(ics) else None, "t_ic": t_ic,
                     "smoke": smoke, "pass": bool(ok)})
        out["cells"].append(cell)
    out["pass"] = any(c.get("pass") for c in out["cells"])
    out["flags"] = flags
    return out


# ── 決策建議（只建議，不執行） ─────────────────────────────────────────

def suggestions(sc: list[dict], attr_rows: list[dict], fals: dict, flags: list[dict]) -> list[str]:
    S = []
    if flags:
        S.append(f"⚠ 回測結果冒煙 {len(flags)} 格 —— 先查外洩，其餘全部暫停。")
    early = [r for r in sc if r["model"] == "claude" and r["n_eff_days"] < 4]
    if early:
        S.append("claude 各格有效天數 < 4：任何「準不準」的結論都太早，本週只做歸因與查核器。")
    for r in attr_rows:
        if r.get("model") != "claude":
            continue
        t = r.get("sel_ic_t_adj")
        if t is not None and r.get("n_eff_days", 0) >= 20:
            if t >= 2:
                S.append(f"{r['market']} h{r['horizon']} 選股 IC 折減後 t={t}：有證據。可依 ITERATE.md §分散度 "
                         "把 p 的橫斷面標準差往 0.4×IC 調（一次最多 ×1.5，下一輪再看）。")
            elif t <= -2:
                S.append(f"{r['market']} h{r['horizon']} 選股 IC 顯著為負（t={t}）：判斷用的主導訊號方向反了，"
                         "回頭看 C 段的分類是不是集中在同一種推論。")
        mb, md = r.get("market_bias"), r.get("market_days", 0)
        if mb is not None and md >= 20 and abs(mb) > 0.05:
            S.append(f"{r['market']} h{r['horizon']} 市場判斷連續偏{'多' if mb > 0 else '空'} {mb:+.3f}"
                     f"（{md} 批）：錨點該檢討，但要對照 attribution 的隱含錨與規則錨，不要憑一段多頭改。")
    if fals.get("batches"):
        sh = fals["shared_text"] / fals["batches"]
        if sh > 0.5:
            S.append(f"否證條件整批共用同一句的批次佔 {sh:.0%}：逐檔寫否證（ITERATE.md §C），否則歸因做不下去。")
        parsed = fals["hit"] + fals["not_hit"]
        if parsed and fals["hit"] / parsed > 0.3:
            S.append(f"否證條件被觸發率 {fals['hit']/parsed:.0%}：觸發卻沒改判斷的，就是場面話。")
    if not S:
        S.append("沒有觸發任何規則：照 WEEKLY.md 做人工歸因，把新型態寫進 LESSONS 與 verify_judgment。")
    return S


def _fmt_sc(sc: list[dict]) -> str:
    L = [f"  {'h':>3} {'市場':<4}{'模型':<12}{'n':>5}{'棄權':>5}{'天':>4}{'有效天':>7}{'準確率':>8}"
         f"{'機率判':>8}{'Brier':>8}{'技能':>8}{'配對差':>8}{'t折減':>7}{'p':>7}"]
    for r in sorted(sc, key=lambda x: (x["horizon"], x["market"], x["brier"])):
        f = lambda v, fmt: "—" if v is None else format(v, fmt)
        L.append(f"  {r['horizon']:>3} {r['market']:<4}{r['model']:<12}{r['n']:>5}{r['n_abstain']:>5}"
                 f"{r['n_days']:>4}{r['n_eff_days']:>7}{f(r['accuracy'], '.1%'):>8}"
                 f"{f(r['accuracy_by_prob'], '.1%'):>8}{f(r['brier'], '.4f'):>8}{f(r['brier_skill'], '+.3f'):>8}"
                 f"{f(r['edge'], '+.3f'):>8}{f(r['edge_t_adj'], '+.2f'):>7}{f(r['p'], '.3f'):>7}")
    return "\n".join(L)


def build(as_of_label: str | None = None) -> dict:
    sc = scorecard()
    a = attr_mod.summary()
    attr_rows = a.get("by_group", [])
    pk = attribution_packet()
    iv = intervals()
    vs = verifier_summary()
    fl = leak_flags()
    acc = acc_mod.summary()
    sug = suggestions(sc, attr_rows, pk["falsifier"], fl)
    rec = {"generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
           "as_of": as_of_label or (max((r["as_of"] for r in pk["worst"]), default=None)),
           "settled_rows": int(score_mod._load().shape[0]) if SETTLEMENTS.exists() else 0,
           "scorecard": sc, "attribution": attr_rows, "intervals": iv,
           "falsifier": pk["falsifier"], "worst": pk["worst"],
           "verifier": vs, "leak_flags": fl, "suggestions": sug,
           "decisions": []}        # 人填：本週決定改什麼、為什麼、預期哪個數字會動
    return rec


def render(rec: dict) -> str:
    L = ["", "=" * 100, f"  每週迭代包 · 結算列 {rec['settled_rows']} · 產生於 {rec['generated_at_utc'][:16]}Z",
         "=" * 100, "", "A. 成績單（分市場；t 已依重疊折減；有效天數 < 4 不出 p）", _fmt_sc(rec["scorecard"]),
         "", "B. 判斷拆解（市場判斷 vs 選股判斷）"]
    def _v(x, fmt):          # DataFrame.to_dict 會把 None 變成 NaN
        return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)
    for r in rec["attribution"]:
        L.append(f"  h{r['horizon']} {r['market']} {r['model']:<10} 錨點偏差 {_v(r.get('market_bias'), '+.3f')}"
                 f"（{r.get('market_days', 0)} 批，來源 {r.get('anchor_source')}） 選股 IC "
                 f"{_v(r.get('sel_rank_ic'), '+.3f')} t折減 {_v(r.get('sel_ic_t_adj'), '+.2f')}"
                 f" 有效天 {r.get('n_eff_days')}")
    L += ["", f"C. 錯誤歸因包：claude 錯最大的 {len(rec['worst'])} 檔（請逐檔填 classify_as：事實錯／推論錯／市場不理會）"]
    for w in rec["worst"]:
        L.append(f"  {w['as_of']} h{w['horizon']} {w['code']:<6} p={w['prob_up']:.3f} 實際 {w['actual_return']:+.1%}"
                 f"  論點：{w['thesis'][:70]}…")
        L.append(f"        否證：{w['falsifier'][:80] or '（無）'}")
    f = rec["falsifier"]
    if f:
        L.append(f"  否證條件機械判定：觸發 {f.get('hit', 0)}／未觸發 {f.get('not_hit', 0)}／待定 {f.get('pending', 0)}"
                 f"／判不了 {f.get('unparseable', 0)}；整批共用同一句的批次 {f.get('shared_text', 0)}/{f.get('batches', 0)}")
    L += ["", "D. 區間覆蓋（名目 80%，看 n 與同日相關，未達 8 個獨立日不可判讀）"]
    for r in rec["intervals"]:
        L.append(f"  h{r['horizon']} {r['market']} {r['model']:<10} 覆蓋 {r['coverage_80']:.1%}（n={r['n']}）")
    L += ["", f"E. 查核器：{json.dumps(rec['verifier'], ensure_ascii=False)}",
          "", f"F. 外洩守門：{'沒有煙' if not rec['leak_flags'] else rec['leak_flags']}",
          "", "G. 建議（只建議，決定要寫進 decisions 並 append）"]
    L += [f"  · {s}" for s in rec["suggestions"]]
    return "\n".join(L)


def main(argv: list[str]) -> None:
    if "--gate" in argv:
        path = argv[argv.index("--gate") + 1]
        name = argv[argv.index("--name") + 1] if "--name" in argv else Path(path).stem
        res = gate(pd.read_parquet(path), name)
        print(json.dumps(res, ensure_ascii=False, indent=1))
        sys.exit(0 if res["pass"] else 1)
    rec = build()
    print(render(rec))
    ITER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ITER_LOG.open("a", encoding="utf-8") as f:            # append-only
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    (OUT_DIR / f"{stamp}.md").write_text("```\n" + render(rec) + "\n```\n", encoding="utf-8")
    print(f"\n已 append：{ITER_LOG}；可讀版：{OUT_DIR / (stamp + '.md')}")


if __name__ == "__main__":
    main(sys.argv[1:])
