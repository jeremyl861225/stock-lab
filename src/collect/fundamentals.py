"""一年期判斷用的基本面長料：財報三表 ＋ 估值 ＋ 月營收，回溯 10 年。

為什麼要 10 年而不是沿用面板的 2.1 年：
  一年期預測的統計基礎是「這檔股票走過幾次完整的產業循環」。
  2.1 年在 250 日尺度上每檔只有約 2 筆獨立觀測，等於沒有樣本。
  2016 年起有 42 季，涵蓋 2018 中美貿易戰、2020 疫情、2022 記憶體與
  面板下行循環、2023–2025 AI 上行 —— 多空都看得到。

為什麼短期判斷不用這些：
  季報一季才換一次值，在 5／20 日尺度上是常數。第零節的靜態雙胞胎測試
  已經證明常數型特徵會把 IC 灌水成「選股」而非「擇時」。
"""
from __future__ import annotations
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect import finmind, universe

START = "2016-01-01"
KINDS = ["fin", "bs", "cf", "per", "rev"]


def backfill(codes: list[str], end: str, refresh: bool = False) -> dict:
    """逐檔逐資料集抓取。回傳 {kind: {ok, fail, rows}}。

    免費層 300 req/hr。50 檔 × 5 個資料集 = 250 個請求，貼著上限，
    所以請求間隔放到 0.8 秒，並且靠 finmind.fetch 的落盤快取避免重抓。
    """
    out = {}
    for kind in KINDS:
        ok = fail = rows = 0
        for c in codes:
            try:
                d = finmind.fetch(kind, c, START, end, refresh=refresh)
                ok += 1; rows += len(d)
            except RuntimeError as e:
                # 額度用盡：立刻收手。繼續跑只會把剩下的請求燒成失敗，
                # 而且下一輪還是得重跑同樣的缺口。
                print(f"\n  ⏸  {e}", flush=True)
                out[kind] = {"ok": ok, "fail": fail, "rows": rows, "quota": True}
                return out
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"  ✗ {kind}/{c}: {type(e).__name__} {str(e)[:60]}", flush=True)
            time.sleep(0.8)
        out[kind] = {"ok": ok, "fail": fail, "rows": rows}
        print(f"  {kind:<5} {ok:>3}/{len(codes)} 檔  {rows:>7,} 筆", flush=True)
    return out


def missing(codes: list[str]) -> dict[str, list[str]]:
    """還沒抓到（或快取區間不夠長）的部分。用來判斷還差多少請求。"""
    from config import RAW
    import json as _json
    out = {}
    for kind in KINDS:
        d = RAW / "finmind" / kind
        need = []
        for c in codes:
            f = d / f"{c}.json"
            if not f.exists():
                need.append(c); continue
            try:
                if _json.loads(f.read_text(encoding="utf-8")).get("start", "9999") > START:
                    need.append(c)
            except Exception:  # noqa: BLE001
                need.append(c)
        if need:
            out[kind] = need
    return out


if __name__ == "__main__":
    import datetime as dt
    end = dt.date.today().isoformat()
    codes = sorted(universe.codes_ever())
    print(f"基本面長料回填：{len(codes)} 檔 × {len(KINDS)} 資料集 = {len(codes)*len(KINDS)} 請求")
    print(f"區間 {START} → {end}\n")
    r = backfill(codes, end)
    print(f"\n完成。失敗合計 {sum(v['fail'] for v in r.values())}")
