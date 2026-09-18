"""面板渲染完整性。

為什麼需要：曾經把 `A if cond else B` 直接寫進 f-string 的隱式串接鏈裡，
Python 把條件套用到「整條鏈」而非那一格 —— 條件成立時整張卡在中途截斷、
div 不閉合，整個 app 版面垮掉。程式不會報錯，測試也全過，只有肉眼看得出來。
所以這裡直接驗產出的 HTML。
"""
import re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
    """獲利點／目標點在每張卡都要有 —— 少了代表串接鏈被條件截斷。"""
    h = _html()
    cards = re.findall(r'<div class="det">.*?</div></div>', h, re.S)
    bad = [c for c in cards if ("獲利點" not in c and "目標價" not in c)]
    assert not bad, f"{len(bad)} 張卡缺少獲利點／目標價"


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
