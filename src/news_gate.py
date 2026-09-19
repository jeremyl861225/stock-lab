# -*- coding: utf-8 -*-
"""新聞閘：判斷「今天有沒有出現足以改寫一年期論點的新聞」。

為什麼需要一道閘，而不是「有新聞就重判」：
  實測 2026-09-18 抓回來的 NVDA 六則新聞，六則全部是目標價與股價回顧
  （"Analyst sets jaw-dropping Nvidia price target"、
    "Nvidia Stock Is Up 21% in Six Months"）。台積電那天六則裡五則是
  外資買賣超與「股價委屈了」。這些每天都有、每天都不一樣。
  若「有新聞就重判」，一年期論點會天天被改寫，而改寫的依據是
  別人對股價的看法 —— 那不是新資訊，那是把雜訊包裝成更新，
  正是 roll_1y 的基礎雜湊要防的同一件事，只是換了一個入口。

所以判準是**事件**，不是**報導**：

  會改寫論點的（實質事件）
    財報與法說、財測調整、併購分拆、擴產建廠與資本支出、
    重大訂單與客戶、法規訴訟制裁關稅、高層異動、產品量產里程碑。

  不會改寫論點的（先擋掉，且優先於上面那組）
    目標價與分析師評等、股價漲跌回顧、法人買賣超、技術分析、
    存股與定期定額、個股推薦清單。
  排除優先的理由：「分析師上修目標價，看好其擴產計畫」同時命中兩組。
  它是一則評等新聞，不是一則擴產新聞 —— 公司沒有做任何新的事。

去重：同一則新聞會連續好幾天出現在 RSS 裡。以正規化標題建立長期記錄，
只有「第一次看到」才算數。少了這一步，一則併購新聞會連續七天觸發重判。
"""
from __future__ import annotations
import datetime as dt, json, re, sys, unicodedata
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA, RAW

SEEN = DATA / "news_seen.json"
KEEP_DAYS = 120      # 「已看過」的保存期。Google News 的查詢窗口是 7 天，
                     # 留 120 天遠超過它，同時讓檔案不會無限長大。

# ── 分工原則 ────────────────────────────────────────────────────────
# 這道閘**不管財報與月營收**。那兩者已經在 roll_1y 的基礎雜湊裡，
# 季報一出、月營收一公告就會觸發重算 —— 新聞再抓一次只會重複標記。
# 新聞要補的是財報看不到、而且不等下一季就會改變論點的事：
#   併購分拆　·　重大投資與建廠　·　財測在財報之間被調整　·
#   法規訴訟制裁　·　經營層異動
# 這個分界也讓閘的召回率可以刻意做低：漏掉一則併購，最慢下一季財報
# 會反映；而每天多送十份「論點待重寫」，會讓所有標記都失去意義。

# ── 先擋：報導的是股價、別人的看法，或只是一篇評論 ──────────────────
EXCLUDE = re.compile(
    # 股價與籌碼
    r"目標價|分析師|評等|上看|喊進|買進評等|賣出評等|投顧|老師|"
    r"大漲|大跌|重挫|飆漲|飆升|狂飆|跌停|漲停|收紅|收黑|開高|開低|"
    r"跌勢|漲勢|均線|法人買賣超|外資買超|外資賣超|投信買超|三大法人|"
    r"融資|當沖|存股|定期定額|殖利率|技術分析|籌碼|股價|填息|除息|"
    # 評論體裁：問句、教學、盤點、週報
    r"[？?]|為何|為什麼|如何|怎麼|該不該|能不能|會不會|值得|"
    r"揭密|揭開|揭露|專家|解析|評析|觀察|盤點|搶先看|懶人包|重點整理|"
    r"一次看|看懂|帶你|教你|討論牆|週報|焦點股|類股|概念股|族群|"
    r"排行|前10大|十大|名單|精選|盤中|盤後|"
    # 推測語氣：還沒發生的事不是事件
    r"可望|有望|拚|挑戰|恐|喊出|預估|評估|傳出|傳聞|市場預期|外媒|外電|"
    # 財報與月營收由基礎雜湊負責，不從新聞進來
    r"月營收|營收公布|自結|營收年增|營收創|財報亮麗|"
    r"price target|analyst|upgrade|downgrade|rating|"
    r"should you buy|stock forecast|stock prediction|best stocks|top stocks|"
    r"stock picks|shares (rise|fall|jump|slip|climb|drop|surge|sink)|"
    r"premarket|pre-market|after hours|what will \$|invested in|"
    r"wall street|here.s what|motley fool|zacks|"
    r"^why |^how |^what |^is |^could |^will |^should ",
    re.I)

# ── 再收：公司或監理機關真的做了一件事，而且做完了 ────────────────────
CATS: list[tuple[str, re.Pattern]] = [
    ("併購分拆", re.compile(
        r"(收購|併購|合併|入股|分拆|分割|出售)[^，。]{0,12}(事業|部門|股權|公司|廠|品牌)|"
        r"(砸|斥資)[^，。]{0,10}(收購|併購|買下|入股)|"
        r"to acquire\b|acquires?\b|agreed? to buy|completes? .{0,15}acquisition|"
        r"merger agreement|spin-?off|divests?\b", re.I)),
    ("重大投資", re.compile(
        r"(斥資|砸|加碼投資|投資)[^，。]{0,12}(億|兆)[^，。]{0,12}(建|蓋|設|擴|廠|產能|基地)|"
        r"動土|破土|新廠[^，。]{0,6}(落成|啟用|投產|量產)|"
        r"breaks? ground|announces? .{0,25}(new (fab|plant|factory)|expansion)|"
        r"\$\d[\d.,]* ?(billion|bn) .{0,25}(invest|plant|fab|expansion)", re.I)),
    ("財測調整", re.compile(
        r"(上修|下修|調升|調降)[^，。]{0,6}(財測|展望|預測|指引|財務預測)|"
        r"(guidance|outlook|forecast) (raised?|cut|lowered?|hiked?)|"
        r"raises? (its )?(full-?year )?(guidance|outlook|forecast)|"
        r"cuts? (its )?(guidance|outlook|forecast)|profit warning", re.I)),
    ("法規訴訟", re.compile(
        r"(遭|被)[^，。]{0,10}(調查|開罰|起訴|控告|禁令|制裁|求償)|"
        r"(裁定|判決|和解)[^，。]{0,8}(賠|罰|敗訴|勝訴)|"
        r"出口管制|禁售|實體清單|反壟斷|反托拉斯|"
        r"antitrust (suit|probe|fine|case)|sued by|files? (a )?lawsuit|"
        r"fined \$|export (ban|control|restriction)|entity list|sanctions? on", re.I)),
    ("經營層異動", re.compile(
        r"(執行長|董事長|總經理|財務長)[^，。]{0,10}(請辭|辭任|接任|上任|異動|換人|去職|閃辭)|"
        r"(ceo|cfo|chief executive) .{0,10}(steps? down|stepping down|to leave)|"
        r"resigns? as|named? (the )?(new )?(ceo|cfo)|appoints? .{0,20}(ceo|cfo)", re.I)),
]

# 明確到不可能是評論的四類：公司或監理機關做了一件已完成的事。
# 這四類可以**越過排除規則** —— 否則「台積電下修全年財測，預估營收成長
# 由 30% 降至 25%」會因為句中有「預估」二字被當成分析師推測擋掉，
# 而它其實是公司自己改了財測，是本閘最該抓到的那種新聞。
# 「重大投資」刻意不在其中：它的用字與產業評論高度重疊，越過排除會破功。
STRONG = {"併購分拆", "財測調整", "法規訴訟", "經營層異動"}


def _norm(t: str) -> str:
    """正規化標題供去重：去掉來源尾綴、全形空白、標點與大小寫差異。"""
    t = unicodedata.normalize("NFKC", t)
    t = re.sub(r"\s+-\s+[^-]{2,40}$", "", t)      # " - Yahoo Finance"
    t = re.sub(r"[^\w一-鿿]+", "", t).lower()
    return t[:120]


def classify(title: str) -> str | None:
    """回傳事件分類；不是實質事件則回 None。

    順序刻意是「強事件 → 排除 → 弱事件」：
      強事件（STRONG）明確到不可能是評論，越過排除規則。
      其餘一律先過排除 —— 「分析師上修目標價，看好其擴產計畫」
      同時命中評等與擴產，它是一則評等新聞，公司什麼都沒做。
    """
    for name, pat in CATS:
        if name in STRONG and pat.search(title):
            return name
    if EXCLUDE.search(title):
        return None
    for name, pat in CATS:
        if pat.search(title):
            return name
    return None


def _load_seen() -> dict:
    if SEEN.exists():
        try:
            return json.loads(SEEN.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _prune(seen: dict, as_of: str) -> dict:
    """丟掉 KEEP_DAYS 之前看過的標題。

    不修剪的話這個檔案會一年長 2MB，而它每天都要進版控。
    可以安全丟掉是因為 RSS 查詢窗口只有 7 天 —— 120 天前的標題
    不可能再出現，留著只是佔空間。
    """
    cut = (dt.datetime.strptime(as_of, "%Y%m%d")
           - dt.timedelta(days=KEEP_DAYS)).strftime("%Y%m%d")
    out = {}
    for code, rec in seen.items():
        kept = {k: v for k, v in rec.items() if v >= cut}
        if kept:
            out[code] = kept
    return out


def scan(as_of: str, codes: list[str] | None = None,
         commit: bool = False) -> dict:
    """掃 as_of 當天的新聞快取，回傳 {code: [實質事件...]}。

    commit=False 時不寫「已看過」記錄 —— 與 roll_1y 同一條原則：
    乾跑若也推進狀態，下一次真正執行時所有新聞都會變成「看過了」。
    """
    d = RAW / "news" / as_of
    if not d.exists():
        return {"as_of": as_of, "available": False, "events": {},
                "note": f"沒有 {as_of} 的新聞快取"}
    seen = _load_seen()
    events: dict[str, list] = {}
    n_items = n_new = 0
    for f in sorted(d.glob("*.json")):
        code = f.stem
        # _market.json 是全市場新聞，不屬於任何一檔。
        # 不排掉的話它會變成一個叫 "_market" 的標的被送進「論點待重寫」。
        if code.startswith("_") or (codes and code not in codes):
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8")).get("payload") or []
        except Exception:  # noqa: BLE001
            continue
        rec = seen.setdefault(code, {})
        hits = []
        for it in payload:
            title = (it.get("title") or "").strip()
            if not title:
                continue
            n_items += 1
            k = _norm(title)
            if k in rec:
                continue
            n_new += 1
            rec[k] = as_of
            cat = classify(title)
            if cat:
                hits.append({"cat": cat, "title": title,
                             "pub": it.get("pub", ""), "link": it.get("link", "")})
        if hits:
            events[code] = hits

    n_codes = len({f.stem for f in d.glob("*.json")
                   if not f.stem.startswith("_")}) or 1
    frac = len(events) / n_codes
    # 一天之內三成以上標的都出現實質事件，最可能的解釋不是三成公司同時
    # 發生大事，而是關鍵字比對出了問題（或有一個全市場事件被逐檔重複報導）。
    # 與其送出 30 份「論點待重寫」，不如說出來讓人看一眼。
    # 比例之外還要有絕對數量門檻。只有三檔的測試情境裡，兩檔命中就是 67%，
    # 但兩則新聞稱不上「異常」—— 沒有下限的比例門檻在小母體上永遠為真。
    flood = frac > 0.30 and len(events) >= 5
    out = {"as_of": as_of, "available": True, "events": events,
           "scanned": n_items, "new": n_new, "codes": n_codes,
           "flagged": len(events), "flood": flood}
    if flood:
        out["note"] = (f"{len(events)}/{n_codes} 檔同日命中（{frac:.0%}）——"
                       f"高於 30% 門檻，先當成比對異常或全市場事件，不逐檔標記重寫")
    if commit and not flood:
        SEEN.write_text(json.dumps(_prune(seen, as_of), ensure_ascii=False),
                        encoding="utf-8")
    return out


if __name__ == "__main__":
    as_of = sys.argv[1] if len(sys.argv) > 1 else None
    if as_of is None:
        import pandas as pd
        p = DATA / "features/panel.parquet"
        as_of = (pd.read_parquet(p, columns=["date"])["date"].max().strftime("%Y%m%d")
                 if p.exists() else dt.date.today().strftime("%Y%m%d"))
    r = scan(as_of, commit="--commit" in sys.argv)
    if not r["available"]:
        print(r["note"])
        sys.exit(0)
    print(f"新聞閘 {as_of}：掃 {r['scanned']} 則、其中 {r['new']} 則是新的，"
          f"{r['flagged']}/{r['codes']} 檔命中實質事件")
    if r.get("note"):
        print(f"  ⚠ {r['note']}")
    for code, hits in sorted(r["events"].items()):
        print(f"  {code}")
        for h in hits:
            print(f"     [{h['cat']}] {h['title'][:78]}")
