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
    run("collect/us.py", allow_fail=True)            # 美股價量／基本面
    run("features/panel.py")                          # 含除權息還原
    run("collect/news_daily.py", allow_fail=True)     # 新聞面（未納入時明天會是空的）
    run("briefing.py")                                # 四面向簡報
    run("predict.py")                                 # 基準線＋統計模型（LLM 無 key 則跳過）
    run("settle.py")                                  # 結算到期預測
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
        key = ("us" if "_us" in f.name else "tw", "h5" if f.name.endswith("h5.json") else "h20")
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
        subprocess.run(["git", "add", *paths], cwd=ROOT)
        msg = f"每日預測 {as_of[:4]}-{as_of[4:6]}-{as_of[6:]}"
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT)
        # GitHub Actions 每天也會 push，不先同步必然 non-fast-forward 被拒
        subprocess.run(["git", "fetch", "origin", "-q"], cwd=ROOT)
        subprocess.run(["git", "rebase", "origin/main"], cwd=ROOT)
        rc = subprocess.run(["git", "push", "origin", "main"], cwd=ROOT).returncode
        print(f"\n{'已提交並推送' if rc == 0 else '！push 失敗（returncode %d），本機已 commit' % rc}：{msg}")
    print("面板：https://jeremyl861225.github.io/stock-lab/")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "finalize": finalize}.get(cmd, prepare)()
