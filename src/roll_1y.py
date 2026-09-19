"""一年期論點的每日滾動（台股＋美股）。

為什麼不能「每天重跑一次判斷」：
  h=250 的日頻預測，相鄰兩天的重疊是 249/250。若每天重算 P漲，
  數字會因為橫斷面百分位的微小擾動而跳動 —— 看起來像今天有了新判斷，
  實際上什麼都沒學到。那是自我欺騙換了一個形式：
  把雜訊包裝成「持續更新」，然後用一年後的結果替它背書。

所以每日滾動嚴格拆成三件事，而且後兩件絕不偽裝成第一件：

  1 重新定價（每天）
    收盤價與 vol_60 每天都在動，目標價、保守價、幅度跟著動。
    論點沒變，只是它在今天的價格上對應到不同的數字。

  2 重新判斷（只在財報輸入真的變了）
    P漲 依據的是季報、月營收（台股）與估值。季報一季換一次、
    月營收一月換一次。在這兩者都沒更新的日子重算 P漲，算出來的差異
    必然是雜訊。判準用「基礎雜湊」：把 P漲 實際依據的輸入湊成一個
    雜湊值，雜湊沒變就沿用昨天的 P漲，並在帳本裡標記 repriced_only。

  3 標記待重寫（財報看不到、但會改寫論點的事）
    併購、重大投資、財測在財報之間被調整、法規訴訟、經營層異動。
    這些由 news_gate 逐日過濾出來（它刻意不管財報與月營收 ——
    那兩者已經在第 2 項裡）。**新聞不會自動改 P漲**：
    財報規則沒有讀新聞的能力，硬套會把一個有根據的數字換成沒根據的。
    新聞只負責指出「這一檔的論點今天需要人重讀」。

第四種情況要特別處理：檢查點被推翻。
  那代表論點的前提倒了，不是「數字微調」而是「這個判斷該重寫」。
  已實查的論點是讀法說會與產業資料寫出來的，程式不該自動改它 ——
  只標記 needs_revision，交回人手重寫。
  規則推導的論點則可以照新資料重算。

兩個市場的節奏本來就不同，不要假裝一樣：
  台股　月營收每月 10 日 ＋ 季報 4 次 → 一年約 16 天會真正重算
  美股　只有季報 4 次（美國公司不公告月營收）→ 一年約 4 天
  所以美股「僅重新定價」的日子佔比更高，那不是系統怠惰，是資訊本來就少。
"""
from __future__ import annotations
import hashlib, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, ROOT, HORIZON_1Y
from features import fundamentals as F
from features import us_fundamentals as UF
from models.rule_1y import prob_up as rule_prob_up
from models.quantiles import quantiles
import checkpoints as CP
import news_gate

JDIR = ROOT / "judgments"
STATE = DATA / "roll_1y_state.json"
SUFFIX = {"TW": "_1y", "US": "_us_1y"}


def _fund(market: str):
    return UF if market == "US" else F


def _latest_judgment(market: str) -> tuple[Path, dict]:
    """取該市場 as_of 最新的一年期判斷檔。

    刻意用內容的 as_of 與 market 欄位而非檔名排序：檔名有三種形態
    （judgment_1y_20260916.json、20260917_1y.json、20260918_us_1y.json），
    字典序會把 judgment_ 開頭的排在數字後面，也會讓 _us_1y 蓋掉 _1y。
    """
    best = None
    for f in JDIR.glob("*_1y*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if "judgments" not in d or (d.get("market") or "TW") != market:
            continue
        if best is None or str(d.get("as_of", "")) > str(best[1].get("as_of", "")):
            best = (f, d)
    if best is None:
        raise RuntimeError(f"找不到 {market} 的一年期判斷檔（judgments/*_1y*.json）")
    return best


def _load_state() -> dict:
    """讀狀態檔，順便把舊格式（只有台股、頂層就是 stocks）搬成分市場格式。"""
    if not STATE.exists():
        return {"markets": {}}
    try:
        d = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {"markets": {}}
    if "markets" in d:
        return d
    return {"markets": {"TW": {"as_of": d.get("as_of"),
                               "stocks": d.get("stocks", {})}}}


def _basis(code: str, snap: pd.DataFrame, per: float | None,
           mrev: pd.DataFrame, market: str) -> str:
    """P漲 實際依據的輸入，湊成一個雜湊。

    刻意只納入「會改變判斷」的東西：
      · 最新已公告季報的季別與各項財務比率
      · 最新已公告月營收的月份與年增率（台股才有）
      · 本益比 —— 它是價格與獲利的比值，價格天天動，
        所以取到小數一位就好，避免每天因為 0.01 的變動就重算。
    刻意不納入：收盤價、vol_60。那兩個歸「重新定價」管；
    也不納入新聞 —— 新聞走 news_gate，它不改數字只標記重寫。
    """
    cols = _fund(market).FUND_COLS
    # 刻意**不**把 market 放進雜湊。狀態檔本來就是分市場存的，放進去是多餘的，
    # 而多餘的那一格會讓所有既有雜湊在改版當天全部失配 ——
    # 下一次滾動會把 38 檔台股全部判成「財報已更新」而重算 P漲，
    # 實際上什麼都沒更新。雜湊的輸入一旦改過，舊基準線就作廢了，
    # 所以這裡只放真正會改變判斷的東西。
    parts = [code]
    r = snap[snap["code"] == code] if not snap.empty else snap
    if not r.empty:
        r = r.iloc[0]
        parts.append(str(pd.Timestamp(r["date"]).date()))
        for c in cols:
            v = r[c]
            parts.append("na" if pd.isna(v) else f"{float(v):.4f}")
    if market == "TW" and not mrev.empty:
        m = mrev[mrev["code"] == code]
        if not m.empty:
            m = m.sort_values("avail_date").iloc[-1]
            parts.append(str(pd.Timestamp(m["date"]).date()))
            for c in ("rev_yoy", "rev_yoy_ttm"):
                v = m[c]
                parts.append("na" if pd.isna(v) else f"{float(v):.4f}")
    parts.append("na" if per is None or pd.isna(per) else f"{float(per):.1f}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def roll(market: str = "TW", as_of: str | None = None,
         commit: bool = False) -> dict:
    """把該市場最新的一年期論點滾到 as_of（預設取面板最後一個交易日）。

    commit=False 是乾跑：算出結果但不動狀態檔與新聞記錄。
    狀態檔記的是「上一次真正發出的預測依據什麼」，若乾跑也寫，
    一次中止的執行就會把基準線往前推，下一次真正滾動時
    所有標的都會被誤判成「輸入未變」—— 那正是這個模組要防的事。
    """
    path, d = _latest_judgment(market)
    pnl = pd.read_parquet(DATA / "features/panel.parquet")
    pnl = pnl[pnl["market"] == market].sort_values(["code", "date"])
    if pnl.empty:
        raise RuntimeError(f"面板沒有 {market} 的資料")
    last_data = pnl["date"].max()
    as_of_ts = pd.Timestamp(as_of) if as_of else last_data
    if as_of_ts > last_data:
        # 這套系統的 as_of 一律是「資料日」而非日曆日（見 daily.py 的 _as_of）。
        # 允許滾到還沒有資料的日期，會產生一個對不到任何收盤的預測，
        # 而它之後要拿什麼價格結算就說不清楚了。
        raise RuntimeError(
            f"as_of {as_of_ts.date()} 超過 {market} 面板最後一個交易日 "
            f"{last_data.date()}；請先跑 collect + features/panel.py")
    as_of_str = as_of_ts.strftime("%Y%m%d")

    pnl = pnl[pnl["date"] <= as_of_ts]
    last = pnl.groupby("code").last()
    pnl["_r"] = pnl.groupby("code")["close"].pct_change()
    vol60 = pnl.groupby("code")["_r"].apply(lambda s: s.tail(60).std())

    snap = _fund(market).as_of(as_of_ts)
    mrev = pd.DataFrame()
    if market == "TW":
        mrev = CP._monthly_rev()
        mrev = mrev[mrev["avail_date"] <= as_of_ts] if not mrev.empty else mrev
    b = pd.read_parquet(DATA / "briefing.parquet")
    b = b[b["market"] == market] if "market" in b.columns else b
    pers = dict(zip(b["code"], b["PER"])) if "PER" in b.columns else {}

    # 新聞閘。乾跑不推進「已看過」記錄，理由與狀態檔相同。
    news = news_gate.scan(as_of_str, codes=[j["code"] for j in d["judgments"]],
                          commit=commit)
    nevents = news.get("events", {}) if not news.get("flood") else {}

    prev = _load_state()["markets"].get(market, {}).get("stocks", {})

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

        cur_basis = _basis(code, snap, pers.get(code), mrev, market)
        pj = prev.get(code, {})
        base_basis = pj.get("basis")
        p_up = float(pj.get("prob_up", j["prob_up"]))
        skew = float(pj.get("skew", (j["up_magnitude"] + j["dn_magnitude"])
                     / (j["up_magnitude"] - j["dn_magnitude"])))
        thesis_as_of = pj.get("thesis_as_of", d["as_of"])

        score = CP.score_all(j, as_of_ts, market)
        was_broken = int(pj.get("broken", -1))
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
                note = "財報已更新；此為已實查論點，待人工複核"
                revise.append({"code": code, "researched": True,
                               "reason": "基礎更新（財報或月營收）",
                               "broken": score["broken"], "verdict": score["verdict"]})
            elif code in rule_p:
                new_p, new_why = rule_p[code]
                if abs(new_p - p_up) > 1e-9:
                    note = f"財報已更新，P漲 {p_up:.3f}→{new_p:.3f}"
                else:
                    note = "財報已更新，重算後 P漲 未變"
                p_up, why = new_p, new_why
                # 偏度沿用 P漲 相對錨點的偏離，與建置時同一條式子
                skew = round((new_p - 0.55) * 1.2, 3)
            else:
                note = "財報已更新，但財報不足以重算，沿用原判斷"
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

        # 新聞事件：只標記，不動 P漲。標記要能被追溯到是哪一則，
        # 否則下次看到「待重寫」也不知道當初為什麼。
        ev = nevents.get(code) or []
        if ev:
            cats = "、".join(sorted({e["cat"] for e in ev}))
            revise.append({"code": code, "researched": bool(j.get("researched")),
                           "reason": f"新聞事件：{cats}",
                           "headlines": [e["title"] for e in ev][:3],
                           "broken": score["broken"], "verdict": score["verdict"]})
            note += f"；出現{cats}新聞，論點待重讀"

        close = float(last.loc[code, "close"])
        base = 0.85 * float(v) * math.sqrt(HORIZON_1Y)
        up, dn = base * (1 + skew), -base * (1 - skew)
        ev_ret = p_up * up + (1 - p_up) * dn
        sig = float(v) * math.sqrt(HORIZON_1Y)
        q10, q90 = quantiles(ev_ret, sig)
        out.append({
            **{k: j[k] for k in ("code", "thesis", "facts", "inference",
                                 "falsifier", "conviction", "checkpoints",
                                 "researched")},
            # stance 必須跟著重算後的 ev 走。沿用舊值會在重新定價讓 ev 跨過 0 時
            # 產生「p>0.5、exp_ret>0 卻標 bearish」的矛盾，並讓 finalize 整條中止
            # （2026-09-19 實際發生於 2395／4958／3443）。棄權不因定價而改變。
            "stance": j["stance"] if j["stance"] == "abstain"
                      else ("bullish" if ev_ret >= 0 else "bearish"),
            "market": market,
            "rationale": why,
            "prob_up": round(p_up, 4),
            "up_magnitude": round(up, 4), "dn_magnitude": round(dn, 4),
            "exp_ret": round(ev_ret, 6),
            "ret_q10": round(q10, 6), "ret_q90": round(q90, 6),
            # 帳本要能分辨「新判斷」與「同一個判斷換了價格」。
            # 少了這兩欄，250 筆重疊預測看起來就像 250 次獨立下注。
            "thesis_as_of": thesis_as_of,
            "repriced_only": cur_basis == base_basis and not ev,
            "basis": cur_basis, "roll_note": note,
            "news_flag": [e["cat"] for e in ev] or None,
            "close_at_call": round(close, 4),
        })

    res = {"as_of": as_of_str, "horizon": HORIZON_1Y, "market": market,
           "version": d.get("version", "1y-1.0.0"),
           "market_context": d.get("market_context", ""),
           "rolled_from": path.name,
           "judgments": out}
    res["_summary"] = {"total": len(out), "repriced_only": repriced,
                       "basis_changed": changed, "needs_revision": revise,
                       "news": {k: v for k, v in news.items() if k != "events"}}
    if not commit:
        return res
    st = _load_state()
    st["markets"][market] = {
        "as_of": as_of_str,
        "stocks": {x["code"]: {"basis": x["basis"], "prob_up": x["prob_up"],
                               "skew": round((x["up_magnitude"] + x["dn_magnitude"])
                                             / (x["up_magnitude"] - x["dn_magnitude"]), 6),
                               "thesis_as_of": x["thesis_as_of"],
                               "broken": CP.score_all(x, as_of_ts, market)["broken"]}
                   for x in out}}
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


def _run_one(market: str, want: str | None) -> int:
    try:
        r = roll(market, want)
    except RuntimeError as e:
        print(f"[{market}] 跳過：{e}")
        return 0
    src_as_of = json.loads((JDIR / r["rolled_from"]).read_text(
        encoding="utf-8"))["as_of"]
    # 滾到跟來源同一天不是滾動，是覆寫。原始判斷檔是那天實際下的注，
    # 覆寫它等於把事後補的欄位混進已鎖定的紀錄。
    if r["as_of"] == src_as_of:
        # 不能滾不代表不用看新聞。今天剛寫好的判斷，當天照樣可能出現
        # 一則改寫它的併購新聞 —— 不印出來，那則新聞就等於沒抓過。
        n = r["_summary"]["news"]
        ev = r["_summary"]["needs_revision"]
        print(f"[{market}] as_of {r['as_of']} 與來源 {r['rolled_from']} 同日，無可滾動"
              + (f"；新聞閘掃 {n['scanned']} 則，命中 {n['flagged']} 檔"
                 if n.get("available") else "；無新聞資料"))
        for x in ev:
            print(f"        {x['code']:<6}{x['reason']}")
            for h in x.get("headlines", []):
                print(f"           · {h[:76]}")
        return 0
    r = roll(market, want, commit=True)
    s = r.pop("_summary")
    out = JDIR / f"{r['as_of']}{SUFFIX[market]}.json"
    out.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{market}] 一年期滾動 {r['as_of']}：{s['total']} 檔 → {out.name}")
    print(f"      僅重新定價 {s['repriced_only']} 檔"
          f"　·　財報更新重算 {len(s['basis_changed'])} 檔")
    n = s["news"]
    if n.get("available"):
        print(f"      新聞閘：掃 {n['scanned']} 則／新 {n['new']} 則，"
              f"命中實質事件 {n['flagged']} 檔"
              + (f"　⚠ {n['note']}" if n.get("note") else ""))
    else:
        print(f"      新聞閘：{n.get('note', '無資料')}")
    if s["basis_changed"]:
        print(f"        {s['basis_changed'][:10]}")
    if s["needs_revision"]:
        print(f"      ⚠ {len(s['needs_revision'])} 項待人工重讀：")
        for x in s["needs_revision"]:
            tag = "已實查（需人工重寫）" if x["researched"] else "規則推導"
            print(f"        {x['code']:<6}{x['verdict']}　已推翻 {x['broken']} 條　"
                  f"{x['reason']}　{tag}")
            for h in x.get("headlines", []):
                print(f"           · {h[:76]}")
    return len(out.read_bytes())


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    want = args[0] if args else None
    markets = ["TW", "US"]
    for mk in markets:
        _run_one(mk, want)
