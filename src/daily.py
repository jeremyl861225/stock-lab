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
    run("briefing.py")                                # 四面向簡報
    run("predict.py")                                 # 基準線＋統計模型（LLM 無 key 則跳過）
    run("settle.py")                                  # 結算到期預測
    subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=ROOT)
    print("\n═══ 準備完成 ═══")
    print("下一步：讀 data/briefing.parquet 做判斷，寫成 judgments/ 下的建構檔，")
    print("       再執行  python src/daily.py finalize")


def finalize(push: bool = True) -> None:
    print("═══ 收尾階段 ═══")
    today = dt.date.today().strftime("%Y%m%d")
    js = sorted((ROOT / "judgments").glob(f"{today}*.json"))
    if not js:
        print(f"！找不到 judgments/{today}*.json —— 判斷尚未產生，中止")
        sys.exit(1)
    run("ingest_judgment.py", *[str(p) for p in js])
    run("panel.py")
    run("ranking.py")
    ok = subprocess.run([PY, "-m", "pytest", "tests/", "-q"], cwd=ROOT).returncode == 0
    if not ok:
        print("！測試未通過，不提交")
        sys.exit(1)
    if push:
        subprocess.run(["git", "add", "-A"], cwd=ROOT)
        msg = f"每日預測 {dt.date.today():%Y-%m-%d}"
        subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT)
        subprocess.run(["git", "push", "origin", "main"], cwd=ROOT)
        print(f"\n已提交並推送：{msg}")
    print("面板：https://jeremyl861225.github.io/stock-lab/")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    {"prepare": prepare, "finalize": finalize}.get(cmd, prepare)()
