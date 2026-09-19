"""一年期定價的不變式（models/price_1y.py）。

存在的理由：2026-09-19 發現一年期的 P漲、期望值、區間來自三套互不相容的算法，
台股 38 檔手寫 P漲 均值 0.552、自身區間隱含的 P漲 只有 0.406，最大差 0.206。
那個錯誤不會拋例外、不會讓面板垮，只會讓 Brier 與區間覆蓋率打分兩個不同的預測。
這一組守的是：**一年期的每一個數字都來自同一個分布**。
"""
import json, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from models.price_1y import price, sigma_daily, implied_prob_up, W_SHORT, LR_MIN, LR_MAX
from models.quantiles import Z10

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("p", [0.30, 0.45, 0.50, 0.55, 0.63, 0.80, 0.90])
@pytest.mark.parametrize("sig", [0.10, 0.30, 0.60, 0.90, 1.20])
def test_all_numbers_come_from_one_distribution(p, sig):
    r = price(p, sig)
    # 分位數反推的 P漲 必須等於輸入的 P漲 —— 這正是 2026-09-19 抓錯用的檢驗
    assert abs(implied_prob_up(r["q10"], r["q90"]) - p) < 1e-9
    # 同一個分布的分位數必然有序
    assert r["q10"] < r["median"] < r["q90"]
    assert r["dn"] < r["median"] < r["up"]
    assert r["q10"] > -1.0 and r["dn"] > -1.0
    # q10（約第 8.7 百分位，因 Z10=1.36 比 1.2816 寬）與 dn（第 (1−p)/2 百分位）
    # 在 p > 0.826 時交叉。實務上 P漲 不會到那裡，但這裡把界線釘住，
    # 免得日後有人看到「保守價高於下跌情境」以為是 bug 而去改數學。
    if p < 0.82:
        assert r["q10"] < r["dn"], f"p={p} 時 q10 應低於 dn"
    if p > 0.84:
        assert r["q10"] > r["dn"], f"p={p} 時兩者應已交叉"
    # 中位數的正負與 P漲 一致 —— 看多／看空的標籤永遠跟得上機率
    assert (r["median"] > 0) == (p > 0.5)
    if p == 0.5:
        assert abs(r["median"]) < 1e-12
    # 平均數與條件平均數守恆等式（全機率公式）
    assert abs(p * r["up_mean"] + (1 - p) * r["dn_mean"] - r["mean"]) < 1e-9
    assert r["mean"] >= r["median"]           # 對數常態右偏


def test_monotone_in_p_and_sigma():
    for sig in (0.2, 0.6):
        meds = [price(p, sig)["median"] for p in (0.4, 0.5, 0.6, 0.7)]
        assert meds == sorted(meds)
    for p in (0.45, 0.6):
        w = [price(p, s)["q90"] - price(p, s)["q10"] for s in (0.2, 0.4, 0.8)]
        assert w == sorted(w), "σ 越大區間必須越寬"


def test_sigma_blend_shrinks_a_regime_spike():
    """近 60 日波動衝到長期的兩倍時，混合後的 σ 必須落在兩者之間。"""
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.015, 900)
    spike = rng.normal(0, 0.030, 60)
    s = sigma_daily(np.concatenate([calm, spike]))
    v60, vlr = spike.std(ddof=1), np.concatenate([calm, spike]).std(ddof=1)
    assert vlr < s < v60
    expect = math.sqrt(W_SHORT * v60 ** 2 + (1 - W_SHORT) * vlr ** 2)
    assert abs(s - expect) < 1e-12
    # 資料不足 LR_MIN 日就不混合，退回 vol_60；不足 60 日回 None
    short = rng.normal(0, 0.02, LR_MIN - 1)
    assert abs(sigma_daily(short) - short[-60:].std(ddof=1)) < 1e-12
    assert sigma_daily(short[:59]) is None
    assert LR_MAX > LR_MIN


def _effective_1y() -> pd.DataFrame:
    p = ROOT / "data/predictions.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d = d.sort_values("created_at_utc").drop_duplicates(
        ["as_of", "horizon", "model", "code"], keep="last")
    return d[(d["model"] == "claude") & (d["horizon"] >= 250)]


def test_ledger_1y_records_are_coherent():
    """帳本裡現行的一年期紀錄，區間隱含的 P漲 必須等於寫下的 P漲。

    只檢查 2026-09-19 之後寫入的列 —— 之前的是兩點模型，依憲法保留不改。
    """
    d = _effective_1y()
    if d.empty:
        return
    d = d[d["created_at_utc"] >= "2026-09-19T05:00:00"]
    if d.empty:
        return
    gap = [abs(implied_prob_up(q10, q90) - p)
           for q10, q90, p in zip(d["ret_q10"], d["ret_q90"], d["prob_up"])]
    bad = d[np.array(gap) > 2e-3]
    assert bad.empty, (f"{len(bad)} 筆一年期紀錄的區間與 P漲 不一致："
                       f"{bad[['as_of','code','prob_up','ret_q10','ret_q90']].head(3).to_dict('records')}")
    assert (np.sign(d["exp_ret"]) == np.sign(d["prob_up"] - 0.5)).all(), \
        "一年期的 exp_ret（中位數）正負與 P漲 不一致"
    assert d["sigma_annual"].notna().all(), "一年期紀錄缺 sigma_annual"


def test_quartiles_are_ordered_with_the_rest():
    for p in (0.35, 0.5, 0.62):
        for sig in (0.15, 0.5, 0.9):
            r = price(p, sig)
            assert r["q10"] < r["q25"] < r["median"] < r["q75"] < r["q90"]
            assert r["q25"] > -1.0


def test_panel_one_year_cards_show_judgment_not_volatility():
    """一年期卡片必須顯示中位價與五成區間，不是目標價／保守價。

    2026-09-19 換掉的理由：目標價與 σ 的 Spearman 是 +0.972、與 P漲 只有 +0.045，
    把所有標的的 P漲 換成同一個值，目標價的橫斷面差異只掉 1%。
    那格數字有 97% 是波動度的讀數。中位價則是判斷：
    中位價高於收盤 ⟺ P漲 > 50%。
    """
    import re
    f = ROOT / "docs/index.html"
    if not f.exists():
        pytest.skip("尚無面板")
    h = f.read_text(encoding="utf-8")
    if 'id="vTW250"' not in h:
        pytest.skip("面板沒有一年期分頁")
    seg = h.split('id="vTW250"')[1].split('<div class="view"')[0]
    assert "中位價" in seg and "五成區間" in seg, "一年期卡片沒有換成中位價／五成區間"
    assert "獲利點" not in seg and "停損點" not in seg, "一年期不該出現可執行價位的字樣"
    # 5／20 日分頁必須維持原樣
    seg20 = h.split('id="vTW20"')[1].split('<div class="view"')[0]
    assert "獲利點" in seg20 and "停損點" in seg20, "5／20 日的價位被誤改"
    assert "中位價" not in seg20


def test_median_price_sits_on_the_right_side_of_close():
    """中位價高於收盤 ⟺ P漲 > 50%。這是這一格之所以是判斷的原因。"""
    import json as _json
    f = ROOT / "judgments/20260918_1y.json"
    if not f.exists():
        pytest.skip("尚無一年期判斷")
    for j in _json.loads(f.read_text(encoding="utf-8"))["judgments"]:
        if not j.get("sigma_annual"):
            continue
        r = price(j["prob_up"], j["sigma_annual"])
        assert (r["median"] > 0) == (j["prob_up"] > 0.5), \
            f"{j['code']} 中位價方向與 P漲 不一致"
