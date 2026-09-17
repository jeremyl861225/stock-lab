"""一年期論點的每日滾動。

為什麼不能「每天重跑一次判斷」：
  h=250 的日頻預測，相鄰兩天的重疊是 249/250。若每天重算 P漲，
  數字會因為橫斷面百分位的微小擾動而跳動 —— 看起來像今天有了新判斷，
  實際上什麼都沒學到。那是自我欺騙換了一個形式：
  把雜訊包裝成「持續更新」，然後用一年後的結果替它背書。

所以每日滾動嚴格拆成兩件事，而且第二件絕不偽裝成第一件：

  重新定價（每天）
    收盤價與 vol_60 每天都在動，目標價、保守價、幅度跟著動。
    論點沒變，只是它在今天的價格上對應到不同的數字。

  重新判斷（只在輸入真的變了）
    P漲 依據的是季報、月營收與估值。季報一季換一次、月營收一月換一次。
    在這兩者都沒更新的日子重算 P漲，算出來的差異必然是雜訊。
    判準用「基礎雜湊」：把 P漲 實際依據的輸入湊成一個雜湊值，
    雜湊沒變就沿用昨天的 P漲，並在帳本裡標記 repriced_only。

第三種情況要特別處理：檢查點被推翻。
  那代表論點的前提倒了，不是「數字微調」而是「這個判斷該重寫」。
  已實查的論點（2330 等）是我讀法說會與產業資料寫出來的，
  程式不該自動改它 —— 只標記 needs_revision，交回人手重寫。
  規則推導的論點則可以照新資料重算。
"""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, ROOT, HORIZON_1Y
from features import fundamentals as F
from models.rule_1y import prob_up as rule_prob_up
import checkpoints as CP

JDIR = ROOT / "judgments"
STATE = DATA / "roll_1y_state.json"


def _latest_judgment() -> tuple[Path, dict]:
    """取 as_of 最新的一年期判斷檔。

    刻意用內容的 as_of 而非檔名排序：檔名有兩種形態
    （judgment_1y_20260916.json 與滾動產出的 20260917_1y.json），
    字典序會把 judgment_ 開頭的排在數字後面，永遠選到最舊的那份。
    """
    best = None
    for f in JDIR.glob("*_1y*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "judgments" not in d:
            continue
        if best is None or str(d.get("as_of", "")) > str(best[1].get("as_of", "")):
            best = (f, d)
    if best is None:
        raise RuntimeError("找不到一年期判斷檔（judgments/*_1y*.json）")
    return best


def _basis(code: str, snap: pd.DataFrame, per: float | None,
           mrev: pd.DataFrame) -> str:
    """P漲 實際依據的輸入，湊成一個雜湊。

    刻意只納入「會改變判斷」的東西：
      · 最新已公告季報的季別與各項財務比率
      · 最新已公告月營收的月份與年增率
      · 本益比 —— 它是價格與獲利的比值，價格天天動，
        所以取到小數一位就好，避免每天因為 0.01 的變動就重算。
    刻意不納入：收盤價、vol_60。那兩個歸「重新定價」管。
    """
    parts = [code]
    r = snap[snap["code"] == code]
    if not r.empty:
        r = r.iloc[0]
        parts.append(str(pd.Timestamp(r["date"]).date()))
        for c in F.FUND_COLS:
            v = r[c]
            parts.append("na" if pd.isna(v) else f"{float(v):.4f}")
    m = mrev[mrev["code"] == code]
    if not m.empty:
        m = m.sort_values("avail_date").iloc[-1]
        parts.append(str(pd.Timestamp(m["date"]).date()))
        for c in ("rev_yoy", "rev_yoy_ttm"):
            v = m[c]
            parts.append("na" if pd.isna(v) else f"{float(v):.4f}")
    parts.append("na" if per is None or pd.isna(per) else f"{float(per):.1f}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def roll(as_of: str | None = None, commit: bool = False) -> dict:
    """把最新的一年期論點滾到 as_of（預設取面板最後一個交易日）。

    commit=False 是乾跑：算出結果但不動狀態檔。
    狀態檔記的是「上一次真正發出的預測依據什麼」，若乾跑也寫，
    一次中止的執行就會把基準線往前推，下一次真正滾動時
    所有標的都會被誤判成「輸入未變」—— 那正是這個模組要防的事。
    """
    path, d = _latest_judgment()
    pnl = pd.read_parquet(DATA / "features/panel.parquet")
    pnl = pnl[pnl["market"] == "TW"].sort_values(["code", "date"])
    last_data = pnl["date"].max()
    as_of_ts = pd.Timestamp(as_of) if as_of else last_data
    if as_of_ts > last_data:
        # 這套系統的 as_of 一律是「資料日」而非日曆日（見 daily.py 的 _as_of）。
        # 允許滾到還沒有資料的日期，會產生一個對不到任何收盤的預測，
        # 而它之後要拿什麼價格結算就說不清楚了。
        raise RuntimeError(
            f"as_of {as_of_ts.date()} 超過面板最後一個交易日 {last_data.date()}；"
            f"請先跑 collect + features/panel.py 取得該日資料")
    as_of_str = as_of_ts.strftime("%Y%m%d")

    pnl = pnl[pnl["date"] <= as_of_ts]
    last = pnl.groupby("code").last()
    pnl["_r"] = pnl.groupby("code")["close"].pct_change()
    vol60 = pnl.groupby("code")["_r"].apply(lambda s: s.tail(60).std())

    snap = F.as_of(as_of_ts)
    mrev = CP._monthly_rev()
    mrev = mrev[mrev["avail_date"] <= as_of_ts] if not mrev.empty else mrev
    b = pd.read_parquet(DATA / "briefing.parquet")
    pers = dict(zip(b["code"], b["PER"])) if "PER" in b.columns else {}

    prev = {}
    if STATE.exists():
        prev = json.loads(STATE.read_text(encoding="utf-8")).get("stocks", {})

    # 規則推導的標的，在基礎更新時要用當下的橫斷面重算 P漲。
    # 只算一次，因為百分位是相對整個橫斷面的 —— 逐檔算會得到不同的母體。
    rule_p = {}
    if not snap.empty:
        rd = snap.copy()
        rd["PER"] = rd["code"].map(pers)
        rd = rd[rd["gm_chg_4q"].notna() | rd["eps_yoy"].notna()]
        if len(rd) >= 5:
            pv, wv = rule_prob_up(rd)
            rule_p = {c: (float(pv.loc[i]), str(wv.loc[i]))
                      for i, c in zip(rd.index, rd["code"])}

    out, changed, revise, repriced = [], [], [], 0
    for j in d["judgments"]:
        code = j["code"]
        v = vol60.get(code)
        if v is None or pd.isna(v) or v <= 0 or code not in last.index:
            continue

        cur_basis = _basis(code, snap, pers.get(code), mrev)
        pj = prev.get(code, {})
        base_basis = pj.get("basis")
        p_up = float(pj.get("prob_up", j["prob_up"]))
        skew = float(pj.get("skew", (j["up_magnitude"] + j["dn_magnitude"])
                     / (j["up_magnitude"] - j["dn_magnitude"])))
        thesis_as_of = pj.get("thesis_as_of", d["as_of"])

        score = CP.score_all(j, as_of_ts)
        was_broken = int(pj.get("broken", -1))
        _ = score  # 下方 note 分支會用到推翻數
        newly_broken = was_broken >= 0 and score["broken"] > was_broken

        why = j["rationale"]
        if base_basis is None:
            note = "首次滾動，沿用原判斷"
        elif cur_basis != base_basis:
            changed.append(code)
            thesis_as_of = as_of_str
            if j.get("researched"):
                # 已實查的論點是讀法說會與產業資料寫出來的。
                # 程式沒有那些資訊，硬套財報規則會把一個有根據的判斷
                # 換成一個沒根據的 —— 標記待複核，不自動改。
                note = "財報或月營收已更新；此為已實查論點，待人工複核"
                revise.append({"code": code, "researched": True,
                               "reason": "基礎更新（財報或月營收）",
                               "broken": score["broken"], "verdict": score["verdict"]})
            elif code in rule_p:
                new_p, new_why = rule_p[code]
                if abs(new_p - p_up) > 1e-9:
                    note = f"財報或月營收已更新，P漲 {p_up:.3f}→{new_p:.3f}"
                else:
                    note = "財報或月營收已更新，重算後 P漲 未變"
                p_up, why = new_p, new_why
                # 偏度沿用 P漲 相對錨點的偏離，與建置時同一條式子
                skew = round((new_p - 0.55) * 1.2, 3)
            else:
                note = "財報或月營收已更新，但財報不足以重算，沿用原判斷"
        else:
            note = "輸入未變，僅重新定價"
            repriced += 1

        if newly_broken:
            # 前提倒了就是論點該重寫，不是把數字調一調。
            # 已實查的論點由人重寫；規則推導的由下一次基礎更新自然反映。
            revise.append({"code": code, "researched": bool(j.get("researched")),
                           "reason": "檢查點新增被推翻",
                           "broken": score["broken"], "verdict": score["verdict"]})
            note += "；檢查點新增被推翻，論點待重寫"

        close = float(last.loc[code, "close"])
        base = 0.85 * float(v) * math.sqrt(HORIZON_1Y)
        up, dn = base * (1 + skew), -base * (1 - skew)
        ev = p_up * up + (1 - p_up) * dn
        sig = float(v) * math.sqrt(HORIZON_1Y)
        out.append({
            **{k: j[k] for k in ("code", "stance", "thesis", "facts", "inference",
                                 "falsifier", "conviction", "checkpoints",
                                 "researched")},
            "rationale": why,
            "prob_up": round(p_up, 4),
            "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
            "exp_ret": round(ev, 6),
            "ret_q10": round(ev - 1.36 * sig, 6),
            "ret_q90": round(ev + 1.36 * sig, 6),
            # 帳本要能分辨「新判斷」與「同一個判斷換了價格」。
            # 少了這兩欄，250 筆重疊預測看起來就像 250 次獨立下注。
            "thesis_as_of": thesis_as_of,
            "repriced_only": cur_basis == base_basis,
            "basis": cur_basis, "roll_note": note,
            "close_at_call": round(close, 4),
        })

    res = {"as_of": as_of_str, "horizon": HORIZON_1Y, "market": "TW",
           "version": d.get("version", "1y-1.0.0"),
           "market_context": d.get("market_context", ""),
           "rolled_from": path.name,
           "judgments": out}
    res["_summary"] = {"total": len(out), "repriced_only": repriced,
                       "basis_changed": changed, "needs_revision": revise}
    if not commit:
        return res
    STATE.write_text(json.dumps({
        "as_of": as_of_str,
        "stocks": {x["code"]: {"basis": x["basis"], "prob_up": x["prob_up"],
                               "skew": round((x["up_magnitude"] + x["dn_magnitude"])
                                             / (x["up_magnitude"] - x["dn_magnitude"]), 6),
                               "thesis_as_of": x["thesis_as_of"],
                               "broken": CP.score_all(x, as_of_ts)["broken"]}
                   for x in out}}, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else None
    r = roll(want)                       # 先乾跑，不動狀態
    # 滾到跟來源同一天不是滾動，是覆寫。原始判斷檔是那天實際下的注，
    # 覆寫它等於把事後補的欄位混進已鎖定的紀錄。
    if r["as_of"] == json.loads((JDIR / r["rolled_from"]).read_text(
            encoding="utf-8"))["as_of"]:
        print(f"as_of {r['as_of']} 與來源 {r['rolled_from']} 同日，無可滾動。"
              f"（面板最後一個交易日還沒往前走）")
        sys.exit(0)
    r = roll(want, commit=True)          # 確定要滾了才寫狀態
    s = r.pop("_summary")
    # 檔名要讓 daily.py finalize 的 glob（judgments/{as_of}*.json）抓得到
    out = JDIR / f"{r['as_of']}_1y.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"一年期滾動 {r['as_of']}：{s['total']} 檔 → {out.name}")
    print(f"  僅重新定價 {s['repriced_only']} 檔"
          f"　·　輸入更新重算 {len(s['basis_changed'])} 檔")
    if s["basis_changed"]:
        print(f"    {s['basis_changed'][:10]}")
    if s["needs_revision"]:
        print(f"  ⚠ {len(s['needs_revision'])} 檔的檢查點新增被推翻，論點待重寫：")
        seen = set()
        for x in s["needs_revision"]:
            if x["code"] in seen:
                continue
            seen.add(x["code"])
            tag = "已實查（需人工重寫）" if x["researched"] else "規則推導"
            print(f"    {x['code']:<6}{x['verdict']}　已推翻 {x['broken']} 條　"
                  f"{x['reason']}　{tag}")
