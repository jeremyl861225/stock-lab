# -*- coding: utf-8 -*-
"""每日流程。分兩階段，中間留給人（或 Claude）做判斷。

    python src/daily.py prepare    # 收資料 → 建特徵 → 基準線與統計模型 → 結算 → 產簡報
    <此處由 Claude 讀簡報、查證、寫 judgments/YYYYMMDD_*.py>
    python src/daily.py finalize   # 寫入判斷 → 產生面板 → 測試 → 提交

刻意分兩段的理由：機器能做的（取數、算特徵、結算、計分）全部自動化，
但「判斷」這一段需要查證新聞、交叉驗證數字、發現資料異常 ——
那是這套系統裡唯一無法自動化、也最容易出錯的環節。
"""
from __future__ import annotations
import subprocess, sys, datetime as dt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")


def run(script: str, *args, allow_fail: bool = False) -> bool:
    print(f"\n── {script} {' '.join(args)}")
    r = subprocess.run([PY, str(ROOT / "src" / script), *args], cwd=ROOT)
    if r.returncode != 0:
        print(f"   ✗ 失敗（returncode {r.returncode}）")
        if not allow_fail:
            sys.exit(r.returncode)
        return False
    return True


def prepare() -> None:
    print("═══ 準備階段 ═══")
    run("collect/universe.py", allow_fail=True)      # 台股 universe（市值前 50）
    run("collect/finmind.py", allow_fail=True)       # 台股價量／法人／融資／除權息／估值／營收
    run("collect/us.py", allow_fail=True)            # 美股價量／估值快照
    # 美股季報（yfinance）。一年期判斷與檢查點全靠它 ——
    # collect/us.py 的 .info 只有當下快照，沒有期別也沒有歷史。
    # 內建 20 小時快取，同一天重複觸發不會重抓。
    # SEC 先跑：它是美股季報的主來源（真實申報日 + 15 年歷史）。
    # yfinance 那支仍照跑，負責 IFRS 申報人（TSM）與 SEC 掛掉時的退路。
    run("collect/sec_edgar.py", allow_fail=True)
    # 美股 PIT 成分股：每月一份快照。價格快取超過 5 天會自動重抓，
    # 所以跑每一天都安全，只有跨月那天會真的多出一份。
    run("collect/pit_universe_us.py", allow_fail=True)
    run("collect/us_fundamentals.py", allow_fail=True)
    # 美股財報排程（yfinance）。內建 20 小時快取。台股的事件由法定期限推導，
    # 不需要抓 —— 見 src/events.py。
    run("events.py", "--fetch", allow_fail=True)
    run("features/panel.py")                          # 含除權息還原
    run("collect/news_daily.py", allow_fail=True)     # 新聞面（未納入時明天會是空的）
    run("briefing.py")                                # 四面向簡報
    run("charts.py", allow_fail=True)                 # K 線資料（docs/charts.json）
    run("predict.py")                                 # 基準線＋統計模型（LLM 無 key 則跳過）
    # 一年期滾動（台股＋美股）。多數日子只重新定價（收盤與 vol_60 變了）；
    # 只有財報輸入真的更新的日子才重算 P漲 ——
    # 台股約 16 天／年（月營收 12 ＋ 季報 4），美股約 4 天／年（只有季報）。
    # 新聞閘在這一步內執行：併購、財測調整、法規、經營層異動會標記
    # 「論點待重讀」，但**不會自動改 P漲**（財報規則讀不了新聞）。
    # 已實查的論點一律不自動改，只標記待複核。
    run("roll_1y.py", allow_fail=True)
    run("settle.py")                                  # 結算到期預測
    # 市場判斷 vs 選股判斷分開計分。手寫 p 的橫斷面標準差只有 0.02–0.03，
    # 而錨點是 0.52–0.53 —— 混在一起算，量到的幾乎全是「錨點對不對」，
    # 而錨點一天只有一個決定、選股一天有 50 個，兩者累積證據的速度差 50 倍。
    run("attribution.py", allow_fail=True)
    # 測試失敗必須擋下來 —— 那些測試在擋未來函數、除權息還原不完整、
    # universe 空檔、跨市場 as_of 錯置，每一項都會讓當天的判斷建立在壞資料上。
    if subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=ROOT).returncode != 0:
        print("\n！測試未通過，停止流程。先查清楚再做判斷。")
        sys.exit(1)
    print("\n═══ 準備完成 ═══")
    print(f"   資料最新交易日：{_as_of()}")
    print(f"   判斷檔請命名為 judgments/build_{_as_of()}.py（台股）")
    print(f"                judgments/build_us_{_as_of()}.py（美股）")
    print("下一步：讀 data/briefing.parquet 做判斷，再執行  python src/daily.py finalize")


def _as_of() -> str:
    """判斷檔的日期是「資料的最新交易日」，不是「執行日」。

    早上 7:30 跑的時候，as_of 是前一個交易日（甚至週一要回到上週五）——
    用 date.today() 去找判斷檔必然落空。這個 bug 會讓排程第一次執行就失敗。
    """
    import pandas as pd
    pnl = ROOT / "data/features/panel.parquet"
    if not pnl.exists():
        return dt.date.today().strftime("%Y%m%d")
    return pd.read_parquet(pnl, columns=["date"])["date"].max().strftime("%Y%m%d")


def finalize(push: bool = True) -> None:
    print("═══ 收尾階段 ═══")
    as_of = _as_of()
    # 同一個 as_of 可能有多個版本（v1／v2／v3）。字典序會讓舊版先進，
    # 而 pid 相同時後進的會被丟棄 —— 實測 v2 台股 50 檔全數被靜默丟掉。
    # 改為依 (horizon, 市場) 分組，各取 mtime 最新的一份。
    allj = list((ROOT / "judgments").glob(f"{as_of}*.json"))
    groups: dict[tuple, Path] = {}
    for f in allj:
        # 一年期必須自成一格。原本的鍵只有 (市場, h5/h20)，
        # 20260917_1y.json 會被歸成 ("tw","h20") 而與當天的台股 20 日判斷
        # 撞在同一格，兩份只會活一份 —— 而且是靜默的。
        if "_1y" in f.name:
            key = ("us" if "_us" in f.name else "tw", "h250")
        else:
            key = ("us" if "_us" in f.name else "tw",
                   "h5" if f.name.endswith("h5.json") else "h20")
        if key not in groups or f.stat().st_mtime > groups[key].stat().st_mtime:
            groups[key] = f
    js = sorted(groups.values())
    print(f"   資料最新交易日 {as_of}，{len(allj)} 份判斷檔中取最新的 {len(js)} 份：")
    for f in js:
        print(f"     {f.name}")
    if not js:
        print(f"！找不到 judgments/{as_of}*.json —— 判斷尚未產生，中止")
        sys.exit(1)
    # 判斷進系統前先過品質查核。四輪人工審核抓到的 30+ 個錯誤，
    # 沒有一個是「模型不夠聰明」，全部是沒查證、用錯變數、時間錯置、單邊採證。
    # 這一關把其中可機械化的部分攔在寫入之前。
    print("\n── 判斷品質查核")
    vr = subprocess.run([PY, str(ROOT / "src/verify_judgment.py"), as_of], cwd=ROOT)
    if vr.returncode != 0:
        print("   （查核器執行失敗，繼續但請人工確認）")
    input_note = "   ↑ 若上面有『待確認』項目，請先逐項確認再繼續\n"
    print(input_note)
    run("ingest_judgment.py", *[str(p) for p in js])
    run("panel.py")
    run("ranking.py")
    ok = subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=ROOT).returncode == 0
    if not ok:
        print("！測試未通過，不提交")
        sys.exit(1)
    if push:
        # 用白名單而非 -A：同一個 repo 可能有其他 session 正在改原始碼，
        # git add -A 會把進行中的改動一起掃進「每日預測」這個 commit。
        paths = ["data/predictions.jsonl", "data/settlements.jsonl",
                 "data/reasoning.jsonl", "data/revisions.jsonl",
                 "docs/", "judgments/", "config/universe_latest.json",
                 "config/universe/", "config/universe_us/",
                 "config/universe_us_latest.json", "LESSONS.md"]
        # 只 add 實際存在的路徑。git add 對任何一個不存在的 pathspec 會整批
        # fatal（returncode 128）而**什麼都不暫存** —— 2026-09-18 實際發生：
        # settle 尚未產出過 settlements.jsonl，於是白名單裡的這一項讓整個 add
        # 失敗，接著 commit 無事可提交、rebase 因未暫存的改動而拒絕，
        # 而 push 回報 "Everything up-to-date"（returncode 0），
        # 最後照樣印出「已提交並推送」。整條鏈沒有一個環節出聲。
        existing = [p for p in paths if (ROOT / p).exists()]
        missing = [p for p in paths if p not in existing]
        if missing:
            print(f"   （白名單中尚未存在、略過：{', '.join(missing)}）")
        msg = f"每日預測 {as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
        if subprocess.run(["git", "add", *existing], cwd=ROOT).returncode != 0:
            print("\n！git add 失敗，未提交。判斷已寫入本機資料，但沒有進版本庫。")
            return
        # 沒有任何改動被暫存時就不要 commit —— 否則 commit 會失敗，
        # 而失敗的原因會被後面的 push 掩蓋掉。
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
            print(f"\n沒有需要提交的變更（判斷可能先前已提交）：{msg}")
            return
        if subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT).returncode != 0:
            print(f"\n！git commit 失敗，未推送：{msg}")
            return
        # GitHub Actions 每天也會 push，不先同步必然 non-fast-forward 被拒。
        # rebase 會因為工作目錄有其他 session 未暫存的改動而拒絕執行，
        # 那不是致命問題（push 若能 fast-forward 仍會成功），但要說出來。
        subprocess.run(["git", "fetch", "origin", "-q"], cwd=ROOT)
        if subprocess.run(["git", "rebase", "origin/main"], cwd=ROOT).returncode != 0:
            print("   （rebase 未執行，多半是工作目錄有未暫存的改動；直接嘗試 push）")
        rc = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode
        print(f"\n{'已提交並推送' if rc == 0 else '！push 失敗（returncode %d），本機已 commit' % rc}：{msg}")
    print("面板：https://jeremyl861225.github.io/stock-lab/")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "finalize": finalize}.get(cmd, prepare)()
