# -*- coding: utf-8 -*-
"""每日新聞收集（台股＋美股）。

此前沒有任何流程呼叫 news.py，data/raw/news/ 只有手動抓的那一天 ——
明天新聞面會是空的，而新聞正是抓出「歸因錯誤」的唯一依據（見 LESSONS.md C 節）。
"""
from __future__ import annotations
import datetime as dt, json, sys, time, urllib.parse
from pathlib import Path
import requests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW, DATA
from collect import news as tw_news, universe

FIN_EN = "stock OR earnings OR revenue OR guidance OR analyst OR price target"


def run(as_of: str | None = None) -> dict:
    import pandas as pd
    if as_of is None:
        pnl = DATA / "features/panel.parquet"
        as_of = (pd.read_parquet(pnl, columns=["date"])["date"].max().strftime("%Y%m%d")
                 if pnl.exists() else dt.date.today().strftime("%Y%m%d"))
    out = RAW / "news" / as_of
    out.mkdir(parents=True, exist_ok=True)
    n_tw = n_us = 0

    for c in universe.load()["constituents"]:
        if (out / f"{c['code']}.json").exists():
            n_tw += 1
            continue
        try:
            if tw_news.stock_news(c["code"], c["name"], as_of):
                n_tw += 1
        except Exception:  # noqa: BLE001
            pass
    try:
        tw_news.market_news(as_of)
    except Exception:  # noqa: BLE001
        pass

    try:
        from collect import us as us_mod
        for c in us_mod.load()["constituents"]:
            f = out / f"{c['code']}.json"
            if f.exists():
                n_us += 1
                continue
            q = urllib.parse.quote(f'"{c["name"].split()[0]}" ({FIN_EN}) when:7d')
            items = []
            try:
                r = requests.get(
                    f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en",
                    headers=tw_news.UA, timeout=25)
                if r.status_code == 200:
                    items = tw_news._parse_rss(r.text)[:6]
            except Exception:  # noqa: BLE001
                pass
            f.write_text(json.dumps({"code": c["code"], "as_of": as_of, "payload": items},
                                    ensure_ascii=False), encoding="utf-8")
            n_us += 1 if items else 0
            time.sleep(0.35)
    except Exception as e:  # noqa: BLE001
        print(f"  美股新聞跳過：{e}")

    print(f"新聞（{as_of}）：台股 {n_tw} 檔、美股 {n_us} 檔")
    return {"as_of": as_of, "tw": n_tw, "us": n_us}


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
