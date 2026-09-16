"""歷史回補（併發版）。可中斷續跑：已快取的直接跳過。

價量走個股月資料（小而快），法人買賣超走 T86 逐日（無個股版本）。
併發限 3 並保留間隔，對官方站台保持禮貌。
"""
from __future__ import annotations
import datetime as dt, sys, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collect import twse, universe

_lock = threading.Lock()
_done = {"n": 0, "err": 0}

def _months(start: str, end: str) -> list[str]:
    s = dt.datetime.strptime(start, "%Y%m").date().replace(day=1)
    e = dt.datetime.strptime(end, "%Y%m").date().replace(day=1)
    out = []
    while s <= e:
        out.append(s.strftime("%Y%m"))
        s = (s.replace(day=28) + dt.timedelta(days=7)).replace(day=1)
    return out

def _tick(total: str, t0: float):
    with _lock:
        _done["n"] += 1
        if _done["n"] % 50 == 0:
            print(f"  {_done['n']}/{total} 完成 錯誤 {_done['err']} "
                  f"({time.time()-t0:.0f}s)", flush=True)

def prices(start_ym: str, end_ym: str, codes: list[str]):
    jobs = [(c, m) for c in codes for m in _months(start_ym, end_ym)]
    t0, total = time.time(), str(len(jobs))
    print(f"價量回補：{len(codes)} 檔 × {len(_months(start_ym,end_ym))} 月 = {len(jobs)} 個請求", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_one_price, c, m): (c, m) for c, m in jobs}
        for f in as_completed(futs):
            try: f.result()
            except Exception as e:
                with _lock: _done["err"] += 1
                print(f"  !! {futs[f]}: {e}", flush=True)
            _tick(total, t0)
    print(f"價量完成 {time.time()-t0:.0f}s 錯誤 {_done['err']}", flush=True)

def _one_price(code: str, ym: str):
    twse.stock_month(code, ym)
    time.sleep(0.8)

def institutions(start: str, end: str):
    s = dt.datetime.strptime(start, "%Y%m%d").date()
    e = dt.datetime.strptime(end, "%Y%m%d").date()
    days = []
    d = s
    while d <= e:
        if d.weekday() < 5: days.append(d.strftime("%Y%m%d"))
        d += dt.timedelta(days=1)
    _done["n"] = 0
    t0, total = time.time(), str(len(days))
    print(f"法人回補：{len(days)} 個工作日", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_one_inst, x): x for x in days}
        for f in as_completed(futs):
            try: f.result()
            except Exception as e:
                with _lock: _done["err"] += 1
                print(f"  !! {futs[f]}: {e}", flush=True)
            _tick(total, t0)
    print(f"法人完成 {time.time()-t0:.0f}s 錯誤 {_done['err']}", flush=True)

def _one_inst(ds: str):
    twse.institutional(ds)
    time.sleep(0.8)

if __name__ == "__main__":
    uni = universe.load()
    codes = [c["code"] for c in uni["constituents"]]
    prices("202409", "202609", codes)
    institutions("20240901", "20260916")
    print("ALL DONE", flush=True)
