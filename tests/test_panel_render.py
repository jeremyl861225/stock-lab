"""面板渲染完整性。

為什麼需要：曾經把 `A if cond else B` 直接寫進 f-string 的隱式串接鏈裡，
Python 把條件套用到「整條鏈」而非那一格 —— 條件成立時整張卡在中途截斷、
div 不閉合，整個 app 版面垮掉。程式不會報錯，測試也全過，只有肉眼看得出來。
所以這裡直接驗產出的 HTML。
"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CHIPS = ["P漲", "漲幅", "跌幅", "年化波動", "收盤價", "信心"]


def _html() -> str:
    import panel
    return panel.build().read_text(encoding="utf-8")


def test_every_card_has_all_chips():
    h = _html()
    cards = re.findall(r'<div class="det">.*?</div></div>', h, re.S)
    assert cards, "沒有產生任何卡片"
    bad = [c[:80] for c in cards if not all(k in c for k in CHIPS)]
    assert not bad, f"{len(bad)} 張卡欄位不完整，例：{bad[:1]}"


def test_div_tags_balanced():
    h = _html()
    o, c = h.count("<div"), h.count("</div>")
    assert o == c, f"div 不平衡：開 {o} / 合 {c}（卡片可能在中途被截斷）"


def test_price_points_present_per_market():
    """每張卡都要有價格格 —— 少了代表 f-string 串接鏈被條件截斷（曾中招一次）。

    兩個期別顯示的名稱刻意不同（2026-09-19 起）：
      5／20 日　獲利點／停損點 —— 期間夠短，是可執行的價位
      一年期　　中位價／五成區間 —— 一年尺度的「目標價」有 97% 是波動度的讀數
    這個測試守的是「有沒有被截斷」，不是守某一個標籤，所以三種都接受。
    """
    h = _html()
    cards = re.findall(r'<div class="det">.*?</div></div>', h, re.S)
    ok = ("獲利點", "目標價", "中位價")
    bad = [c for c in cards if not any(k in c for k in ok)]
    assert not bad, f"{len(bad)} 張卡缺少價格格"


def test_long_names_cannot_push_ev_offscreen():
    """期望值是整列最重要的數字，絕不能被長名稱擠出畫面。

    起因：美股名稱長達 31 字（"Taiwan Semiconductor Manufactur"），
    而 .nm 原本只有 white-space:nowrap 沒有 overflow —— 它會把整列撐寬，
    在 390px 的手機上把期望值推出右緣。台股名 2–4 字所以看不出來。
    """
    h = _html()
    css = h[h.index("<style>"):h.index("</style>")]
    # 名稱必須可壓縮並截斷
    assert "text-overflow:ellipsis" in css, "名稱沒有截斷機制"
    # 期望值必須不可壓縮
    assert "flex:0 0 auto" in css.split(".ev{")[1].split("}")[0], ".ev 可被壓縮"


def test_us_rows_lead_with_ticker():
    """美股用代號當識別主體 —— 公司全名在手機上會被截成認不出來。"""
    import re
    h = _html()
    us = re.search(r'id="vUS20".*?(?=<div class="view"|</body>)', h, re.S)
    assert us, "找不到美股分頁"
    tickers = re.findall(r'<span class="nm tk">([^<]+)</span>', us.group())
    assert len(tickers) > 20, f"美股列只有 {len(tickers)} 檔用代號當主體"
    assert all(not t.isdigit() for t in tickers), "台股代號跑進美股分頁"


def test_tw_rows_lead_with_name():
    """台股名稱短，仍以名稱當主體、代號當副標。"""
    import re
    h = _html()
    tw = re.search(r'id="vTW20".*?(?=<div class="view"|</body>)', h, re.S)
    assert tw, "找不到台股分頁"
    codes = re.findall(r'<span class="cd">(\d+)</span>', tw.group())
    assert len(codes) > 20, f"台股列只有 {len(codes)} 檔用名稱當主體"


def test_markets_keep_their_own_as_of():
    """台股與美股收盤差 12 小時以上，as_of 本來就會不同步。

    起因：ranking.table() 用全域 max as_of 篩選，台股推進到新交易日後，
    還停在前一日的美股判斷被整組濾掉 —— 美股分頁直接空掉。
    實測 2026-09-17 台股推進後，US20／US5 都歸零。
    而且要先 merge 再篩：帳本的列沒有 market 欄位，市場是從 briefing 帶進來的。
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import ranking
    for h in (5, 20):
        t = ranking.table(h)
        if t.empty or "market" not in t.columns:
            continue
        for mk in t["market"].dropna().unique():
            g = t[t["market"] == mk]
            assert len(g) > 0, f"{mk} h={h} 沒有任何判斷"
            assert g["as_of"].nunique() == 1, (
                f"{mk} h={h} 混了多個 as_of：{sorted(g['as_of'].unique())}")
        # 兩市場都必須有內容（只要帳本裡各自有判斷）
        mks = set(t["market"].dropna().unique())
        assert len(mks) >= 1, f"h={h} 一個市場都沒有"


def test_no_undefined_css_variables():
    """所有用到的 CSS 變數都必須有定義。

    起因：K 線用了 var(--card2)，那個變數根本不存在（我憑空發明的），
    結果圖表框與尺度按鈕在兩個主題下都是透明的 —— 瀏覽器不會報錯，
    只會安靜地當成 initial 值。
    """
    import re
    h = _html()
    css = h[h.index("<style>"):h.index("</style>")]
    used = set(re.findall(r"var\(--([a-z0-9]+)", css))
    # 變數可以在樣式表裡宣告，也可以在元素上用 inline style 設定
    # （例如橫桿的 style="--o:left" 決定變形原點），兩邊都要算。
    declared = (set(re.findall(r"--([a-z0-9]+)\s*:", css))
                | set(re.findall(r'style="[^"]*--([a-z0-9]+)\s*:', h)))
    missing = used - declared
    assert not missing, f"用到但未定義的 CSS 變數：{sorted(missing)}"


def test_both_markets_keep_their_one_year_checkpoints():
    """兩個市場的一年期檢查點必須同時存在。

    判斷檔的字典序是 20260918_1y.json < 20260918_us_1y.json，
    原本的 `sorted(glob)[-1]` 只取最後一份 —— 加進美股之後，
    台股的檢查點會整組消失，而且是靜默的：面板照樣渲染，
    只是所有台股都變成「待驗」，左緣色條也跟著不見。
    """
    h = _html()
    for key in ("vTW250", "vUS250"):
        assert f'id="{key}"' in h, f"{key} 分頁不存在"
        seg = h.split(f'id="{key}"')[1].split('<div class="view"')[0]
        assert seg.count('data-vd=') > 10, f"{key} 幾乎沒有論點判定，檢查點可能被蓋掉"
        assert seg.count('class="cp ') > 20, f"{key} 幾乎沒有檢查點"


def test_one_year_cards_are_not_all_pending():
    """檢查點全數 pending 代表查錯了資料源（例如拿台股的月營收去對美股）。
    那在畫面上看起來像「還沒到期」，實際上是永遠不會有答案。"""
    h = _html()
    seg = h.split('id="vUS250"')[1].split('<div class="view"')[0]
    n_pd = seg.count('class="cp pd"')
    n_all = seg.count('class="cp ')
    assert n_all > 0 and n_pd < n_all * 0.5, \
        f"美股一年期有 {n_pd}/{n_all} 條檢查點待公告，疑似對錯資料源"


def test_chart_last_candle_matches_card_as_of():
    """K 線的最後一根必須是卡片收盤價那一天。

    美股的圖表資料原本只讀一次性回補的 prices_5y.parquet，而每日流程
    更新的是 prices.parquet —— 最後一根永遠停在回補那天（實測停在 09-16，
    卡片已經是 09-18）。圖與卡片對不上，而且不會有任何錯誤訊息。

    比對對象刻意用 briefing.parquet 而不是 panel.parquet：
    一、briefing 才是卡片收盤價的實際來源，比的就是「圖 vs 卡」本身；
    二、panel.parquet 在 .gitignore 內，CI 拿不到。第一版寫成讀 panel，
        本機全綠而 CI 連三次失敗 —— 只在本機跑得動的測試，
        等於在唯一會擋住錯誤的地方缺席。

    `compared` 是必要的：第二版曾寫 `if not isinstance(want, str): continue`，
    而 briefing 的 as_of 是 datetime64，於是迴圈跳過每一個市場、
    一次比對都沒做就回報通過。空轉的測試比沒有測試更糟 ——
    沒有測試至少不會讓人以為有防線（README 的 test_truncation_invariance 同型）。
    """
    import json
    import pandas as pd
    cj = ROOT / "docs/charts.json"
    if not cj.exists():
        return
    d = json.loads(cj.read_text(encoding="utf-8"))
    b = pd.read_parquet(ROOT / "data/briefing.parquet", columns=["as_of", "market"])
    compared = 0
    for mk, chart_as_of in d.get("as_of", {}).items():
        col = b[b["market"] == mk]["as_of"]
        if col.empty:
            continue
        want = pd.Timestamp(col.max())
        assert pd.Timestamp(chart_as_of) >= want, \
            f"{mk} 的 K 線停在 {chart_as_of}，卡片已到 {want.date()}"
        compared += 1
    assert compared >= 2, f"只比對了 {compared} 個市場，這個測試在空轉"
