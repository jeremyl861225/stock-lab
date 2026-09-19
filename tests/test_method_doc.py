"""METHOD.md 與程式碼的一致性。

為什麼需要這個：文件會漂移，而且漂移不會出聲。
METHOD.md 的用途是「換一個沒有記憶的 session 也能重現這套方法」——
一旦它描述的常數與程式碼不符，它就從資產變成陷阱：
新 session 會照著一份過期的規格去改判斷，而且完全不知道自己在偏離。

所以這裡把文件裡出現的每一個數字，逐一釘回它在程式碼裡的出處。
改了程式忘了改文件（或反過來）就會失敗。
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# (在 METHOD.md 裡必須出現的字串, 程式檔, 該檔裡必須匹配的 regex)
PINS = [
    ("Z10 = 1.36",        "src/models/quantiles.py",        r"Z10\s*=\s*1\.36"),
    ("z = 1.2816",        "src/models/statistical.py",      r"z=1\.2816"),
    ("ANCHOR = 0.55",     "src/models/rule_1y.py",          r"ANCHOR\s*=\s*0\.55"),
    ("0.44, 0.64",        "src/models/rule_1y.py",          r"FLOOR,\s*CAP\s*=\s*0\.44,\s*0\.64"),
    ("gm_chg_4q 0.06",    "src/models/rule_1y.py",          r'"gm_chg_4q":\s*0\.06'),
    ("roe_ttm 0.04",      "src/models/rule_1y.py",          r'"roe_ttm":\s*0\.04'),
    ("MIN_TRAIN_ROWS = 2000", "src/config.py",              r"MIN_TRAIN_ROWS\s*=\s*2000"),
    ("固定 **0.55**",     "src/models/baselines.py",        r'"prob_up":\s*0\.55'),
    ("C=0.3",             "src/models/statistical.py",      r"C=0\.3"),
    ("保存 120 天",       "src/news_gate.py",               r"KEEP_DAYS\s*=\s*120"),
    ("≥5 檔命中",         "src/news_gate.py",               r"len\(events\)\s*>=\s*5"),
    ("10-Q 季末後 40 天", "src/features/us_fundamentals.py", r"LAG_Q,\s*LAG_FY\s*=\s*40,\s*60"),
    ("外國發行人 45／90", "src/features/us_fundamentals.py", r"LAG_Q_FPI,\s*LAG_FY_FPI\s*=\s*45,\s*90"),
    ("HORIZON_1Y",        "src/config.py",                  r"HORIZON_1Y\s*=\s*250"),
    # 一年期定價：只有 P漲 與 σ 兩個輸入（2026-09-19 起）
    ("W_SHORT = 0.6",     "src/models/price_1y.py",         r"W_SHORT\s*=\s*0\.6"),
    ("LR_MIN, LR_MAX = 250, 1000", "src/models/price_1y.py", r"LR_MIN,\s*LR_MAX\s*=\s*250,\s*1000"),
    ("price_1y.price()",  "src/roll_1y.py",                 r"price_1y\("),
]


def _doc() -> str:
    return (ROOT / "METHOD.md").read_text(encoding="utf-8")


def test_method_doc_exists():
    assert (ROOT / "METHOD.md").exists(), "METHOD.md 不見了 —— 那是方法的唯一規格書"


def test_documented_constants_match_code():
    doc = _doc()
    bad = []
    for phrase, path, pat in PINS:
        if phrase not in doc:
            bad.append(f"METHOD.md 沒有提到「{phrase}」")
            continue
        src = (ROOT / path).read_text(encoding="utf-8")
        if not re.search(pat, src):
            bad.append(f"「{phrase}」在 {path} 裡對不上（regex: {pat}）")
    assert not bad, "文件與程式碼不同步：\n  " + "\n  ".join(bad)


def test_five_twenty_day_formula_is_pinned_to_latest_build():
    """5／20 日的方法只存在於每天的 build 檔裡，靠複製上一份傳遞。

    這是結構性的脆弱點：沒有任何一支模組持有它。所以至少要確保
    METHOD.md 描述的那兩條式子，在最近一份 build 檔裡真的長那樣。
    """
    builds = sorted((ROOT / "judgments").glob("build_2*.py"))
    builds = [b for b in builds if "_1y" not in b.name]
    assert builds, "找不到任何 5／20 日的 build 檔"
    src = builds[-1].read_text(encoding="utf-8")
    assert re.search(r"math\.sqrt\(horizon\s*/\s*20\)", src), \
        f"{builds[-1].name} 的期間縮放式子與 METHOD.md §4.2 不符"
    assert re.search(r"0\.85\s*\*\s*vol", src), \
        f"{builds[-1].name} 的幅度式子與 METHOD.md §1.3 不符"
    doc = _doc()
    assert "p_h = 0.5 + (p20 − 0.5) × √(h / 20)" in doc
    assert "base = 0.85 × vol × √h" in doc


def test_anchor_values_are_stated_with_a_date():
    """錨點是整批判斷的共同平移項，而且會隨總體環境改變。
    文件裡沒有寫明「哪一天起」的錨點，等於邀請下一個 session 用過期的值。"""
    doc = _doc()
    assert re.search(r"2026-09-18 起", doc), "錨點沒有標明生效日"
    for mk, val in (("台股", "0.53"), ("美股", "0.52")):
        assert re.search(rf"\|\s*{mk}\s*\|\s*\*\*{val}\*\*", doc), \
            f"{mk} 錨點 {val} 未列在 METHOD.md §4.3"
