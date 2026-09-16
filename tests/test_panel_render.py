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
