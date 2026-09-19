"""新聞閘的不變式。

這道閘存在的唯一理由是「不讓一年期論點被別人對股價的看法改寫」。
所以最重要的測試不是「有沒有抓到事件」，而是「有沒有把雜訊當成事件」——
召回率低會漏掉一則併購，最慢下一季財報會反映；
精準率低則會每天送出十份「論點待重寫」，讓所有標記都失去意義。
"""
import json, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import news_gate as NG

# 這些都是真實抓回來的標題（2026-09-16～18），不是杜撰的
NOISE = [
    "Nvidia Stock Forecast: Where Investors Could Stand in 5 Years - Yahoo Finance",
    "Analyst sets jaw-dropping Nvidia price target - thestreet.com",
    "What Will $5,000 Invested in Nvidia Stock Be Worth in 5 Years? - 24/7 Wall St.",
    "Nvidia Stock Is Up 21% in Six Months. Here's What the Street Still Isn't Pricing In",
    "台積電明除息7元！投顧董座看好「秒填息」 他喊：股價委屈了 - Yahoo股市",
    "外資猛砍台積電！4天提款2.39萬張　今日買賣前10大曝光 - ETtoday財經雲",
    "台積電股價續飆？外媒曝「明年底目標價」：真正潛力可能更高 - TVBS新聞網",
    "三大法人買賣超 – 外資買超(2330)台積電、(6770)力積電",
    "不跌了？台達電營收公布後的跌勢反轉 逆勢收復5日線 - 經濟日報",
    "緯創今年營收有亮眼表現 董事長林憲銘做對了什麼？ - 經濟日報",
    "AI伺服器出貨強勁！華碩8月營收歷史次高年增51.17％ - 自由時報",
    "研華(2395)訂單爆滿、產能滿載，為何毛利率卻跌破38%？ - news.cnyes.com",
    "〈財經週報-半導體供應鏈千金股〉PCB翻身 台光電股價超車台積電 - 自由時報",
]

EVENTS = [
    ("華邦電砸356億元收購英飛凌記憶體事業 Spansion品牌復活 - news.cnyes.com", "併購分拆"),
    ("英特爾執行長季辛格請辭 董事會啟動接班程序", "經營層異動"),
    ("台積電下修全年財測 預估營收成長由 30% 降至 25%", "財測調整"),
    ("Broadcom agreed to buy VMware in $61 billion deal", "併購分拆"),
    ("Google 遭歐盟反壟斷開罰 24 億歐元", "法規訴訟"),
]


@pytest.mark.parametrize("title", NOISE)
def test_price_and_commentary_are_never_events(title):
    """目標價、股價回顧、法人買賣超、評論體裁一律不得算成事件。
    這些每天都有、每天都不一樣 —— 放行等於讓論點天天被雜訊改寫。"""
    assert NG.classify(title) is None, f"雜訊被判成事件：{title}"


@pytest.mark.parametrize("title,cat", EVENTS)
def test_real_events_are_caught(title, cat):
    assert NG.classify(title) == cat


def test_exclusion_beats_inclusion():
    """同時命中兩組時，排除優先。
    「分析師上修目標價，看好其擴產計畫」是一則評等新聞，不是擴產新聞 ——
    公司什麼都沒做。"""
    assert NG.classify("分析師上修目標價，看好台積電的擴產計畫") is None


def test_earnings_are_left_to_the_basis_hash():
    """財報與月營收刻意不從新聞進來 —— 它們已經在 roll_1y 的基礎雜湊裡，
    再抓一次只會對同一件事重複標記。"""
    assert NG.classify("台積電公布第二季財報 每股盈餘 27.25 元") is None
    assert NG.classify("鴻海8月營收年增52% 創同期新高") is None


def test_dedupe_by_normalised_title():
    """同一則新聞會連續好幾天出現在 RSS 裡。去重用正規化標題，
    否則一則併購會連續七天觸發重判。"""
    a = "華邦電砸356億元收購英飛凌記憶體事業 Spansion品牌復活 - news.cnyes.com"
    b = "華邦電砸356億元收購英飛凌記憶體事業　Spansion品牌復活 - Yahoo股市"
    assert NG._norm(a) == NG._norm(b)


def test_market_wide_file_is_not_a_stock(tmp_path, monkeypatch):
    """_market.json 是全市場新聞。不排掉的話會冒出一個叫 _market 的標的。"""
    d = tmp_path / "news" / "20260101"
    d.mkdir(parents=True)
    ev = {"title": "華邦電砸356億元收購英飛凌記憶體事業", "pub": "", "link": ""}
    (d / "_market.json").write_text(json.dumps({"payload": [ev]}), encoding="utf-8")
    (d / "2344.json").write_text(json.dumps({"payload": [ev]}), encoding="utf-8")
    monkeypatch.setattr(NG, "RAW", tmp_path)
    monkeypatch.setattr(NG, "SEEN", tmp_path / "seen.json")
    r = NG.scan("20260101")
    assert set(r["events"]) == {"2344"}


def test_dry_run_does_not_persist_seen(tmp_path, monkeypatch):
    """乾跑不得推進『已看過』記錄 —— 與 roll_1y 的狀態檔同一條原則：
    一次中止的執行會讓下一次真正執行時，所有新聞都變成『看過了』。"""
    d = tmp_path / "news" / "20260101"
    d.mkdir(parents=True)
    (d / "2344.json").write_text(json.dumps(
        {"payload": [{"title": "華邦電砸356億元收購英飛凌記憶體事業"}]}),
        encoding="utf-8")
    monkeypatch.setattr(NG, "RAW", tmp_path)
    seen = tmp_path / "seen.json"
    monkeypatch.setattr(NG, "SEEN", seen)
    NG.scan("20260101")
    assert not seen.exists()
    NG.scan("20260101", commit=True)
    assert seen.exists()
    assert NG.scan("20260101")["events"] == {}, "已看過的新聞不該再次觸發"


def test_flood_guard(tmp_path, monkeypatch):
    """一天之內三成以上標的都命中，最可能的解釋是比對出了問題，
    而不是三成公司同時發生大事。與其送出 30 份『待重寫』，不如說出來。"""
    d = tmp_path / "news" / "20260101"
    d.mkdir(parents=True)
    for i in range(20):
        (d / f"C{i}.json").write_text(json.dumps(
            {"payload": [{"title": f"公司{i}砸100億元收購某某事業"}]}),
            encoding="utf-8")
    monkeypatch.setattr(NG, "RAW", tmp_path)
    monkeypatch.setattr(NG, "SEEN", tmp_path / "seen.json")
    r = NG.scan("20260101")
    assert r["flood"] is True
    assert "note" in r
