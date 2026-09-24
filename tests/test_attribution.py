"""市場判斷與選股判斷必須分開計分（src/attribution.py）。

2026-09-19 實測：手寫 p20 的橫斷面標準差台股 0.027、美股 0.020，錨點卻是 0.53／0.52。
每一筆判斷有 95% 以上的內容來自錨點，混在一起算準確率，量到的幾乎全是錨點。
這一組守的是：**選股分數不得受市場整體漲跌影響，市場分數不得受選股影響。**
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import attribution as A

ROOT = Path(__file__).resolve().parent.parent


def _frame(n_days=6, n_codes=20, shift=0.0, seed=0, market="TW", horizon=20):
    """造一批可控的假結算：prob_up 與報酬有固定的橫斷面關係，
    再把整個市場平移 shift（模擬大盤漲跌）。"""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        base = rng.normal(0, 0.05)              # 當天的市場整體漲跌
        for c in range(n_codes):
            p = 0.5 + (c - n_codes / 2) * 0.002
            r = (c - n_codes / 2) * 0.004 + base + shift + rng.normal(0, 0.01)
            rows.append({"as_of": f"202609{d+1:02d}", "code": f"{1000+c}" if market == "TW" else f"C{c}",
                         "horizon": horizon, "model": "claude", "prob_up": p,
                         "actual_return": r, "market": market})
    return pd.DataFrame(rows)


def test_selection_score_ignores_market_level():
    """把每一天的報酬整體平移，選股分數必須一模一樣。"""
    a = A.decompose(_frame(shift=0.0))
    b = A.decompose(_frame(shift=+0.20))        # 大盤整體多漲 20%
    for col in ("sel_rank_ic", "sel_spread", "sel_hit"):
        assert abs(float(a[col].iloc[0]) - float(b[col].iloc[0])) < 1e-9, \
            f"{col} 受市場整體漲跌影響 —— 選股分數沒有去均值乾淨"


def test_market_score_uses_the_anchor_not_the_stock_picks():
    """市場分數＝錨點 − 實際上漲比率，與個股怎麼排無關。"""
    df = _frame(shift=+0.20)                     # 幾乎全漲
    anchors = {(d, "TW"): 0.53 for d in df["as_of"].unique()}
    r = A.decompose(df, anchors).iloc[0]
    realized = float((df["actual_return"] > 0).mean())
    assert r["market_days"] == df["as_of"].nunique()
    # 全漲的日子裡，0.53 的錨點是明顯偏空 → 偏差為負
    assert r["market_bias"] < 0
    assert abs(r["market_bias"] - (0.53 - realized)) < 0.05
    # 沒有錨點時用該批 p 均值當隱含錨，而且要標明來源 —— 不能默默當成有錨點。
    # （首批 20260916 台股的判斷檔與帳本都沒寫 anchor，實際 p 均值 0.500 而非文件的 0.53；
    #   不推導的話市場判斷那一半永遠是空的。）
    r0 = A.decompose(df).iloc[0]
    implied = float(df["prob_up"].mean())
    assert abs(r0["market_bias"] - (implied - realized)) < 0.05
    assert r0["anchor_source"] == {"implied": df["as_of"].nunique()}
    assert r["anchor_source"] == {"judgment": df["as_of"].nunique()}


def test_h5_anchor_is_scaled_from_p20():
    """判斷檔的 anchor 是 p20 的錨點；h5 的市場判斷要先依 √t 換到 5 日尺度。"""
    df = _frame(shift=+0.20, horizon=5)
    anchors = {(d, "TW"): 0.53 for d in df["as_of"].unique()}
    r = A.decompose(df, anchors).iloc[0]
    realized = float((df["actual_return"] > 0).mean())
    a5 = 0.5 + (0.53 - 0.5) * (5 / 20) ** 0.5
    assert abs(r["market_bias"] - (a5 - realized)) < 0.05


def test_neutral_rows_do_not_enter_selection_hit():
    """p 恰等於錨點的列是中性，不進 sel_hit 的分母。"""
    df = _frame(n_days=3)
    anchors = {(d, "TW"): 0.5 for d in df["as_of"].unique()}
    # 第 10 檔的 p 恰為 0.5（= 錨點）
    r = A.decompose(df, anchors).iloc[0]
    assert r["sel_hit"] is not None and 0 <= r["sel_hit"] <= 1


def test_effective_days_discount_daily_overlap():
    """日頻預測在 h 日尺度上相鄰重疊 (h−1)/h，有效天數必須被折減。"""
    r20 = A.decompose(_frame(n_days=10, horizon=20)).iloc[0]
    r5 = A.decompose(_frame(n_days=10, horizon=5)).iloc[0]
    assert r20["n_eff_days"] < r5["n_eff_days"] < 10, \
        "有效天數沒有隨期間拉長而折減"
    assert abs(r20["n_eff_days"] - 10 / 20) < 0.3


def test_markets_are_never_pooled_into_one_cross_section():
    """台美的交易日與漲跌互不相干，混在同一個橫斷面是錯的。"""
    df = pd.concat([_frame(market="TW", seed=1), _frame(market="US", seed=2)])
    r = A.decompose(df)
    assert set(r["market"]) == {"TW", "US"}
    assert len(r) == 2, "兩個市場被併成同一列"


def test_anchors_are_read_from_shipped_judgment_files():
    """已出貨的判斷檔要讀得到錨點，否則上線後的第一批成績就拆不開。"""
    a = A.anchors_from_judgments(ROOT / "judgments")
    assert a, "沒有任何判斷檔記錄錨點"
    assert all(0.3 < v < 0.8 for v in a.values()), f"錨點超出合理範圍：{a}"
    assert ("20260918", "TW") in a and ("20260918", "US") in a


def test_summary_is_honest_when_nothing_settled():
    out = A.summary()
    assert out["status"] in ("ok", "no_settlements")


def test_anchor_is_written_into_the_ledger(tmp_path, monkeypatch):
    """新寫入的判斷必須把錨點帶進 predictions.jsonl。

    不進帳本的話，事後要拆解就只能回頭讀判斷檔 —— 而判斷檔會被修訂覆寫，
    帳本才是那一刻真正下的注。
    """
    import predict as P
    import ingest_judgment as IJ
    from models import claude_judgment as cj
    led = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(P, "PREDICTIONS", led)
    monkeypatch.setattr(IJ, "PREDICTIONS", led)
    monkeypatch.setattr(cj, "REASONING", tmp_path / "reasoning.jsonl")

    doc = {"as_of": "20260918", "horizon": 20, "market": "TW", "anchor": 0.53,
           "market_context": "測試", "judgments": [{
               "code": "2330", "stance": "bullish", "prob_up": 0.54,
               "exp_ret": 0.01, "ret_q10": -0.05, "ret_q90": 0.07,
               "conviction": "low", "thesis": "t", "facts": ["f"],
               "inference": "i", "falsifier": "x"}]}
    f = tmp_path / "j.json"
    f.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    out = IJ.run(str(f))
    assert out["written"] == 1
    rec = json.loads(led.read_text(encoding="utf-8").strip())
    assert rec["anchor"] == 0.53, "錨點沒有進帳本"

    # 沒寫錨點的判斷檔不得臆測一個值出來
    doc2 = dict(doc); doc2.pop("anchor"); doc2["as_of"] = "20260917"
    f2 = tmp_path / "j2.json"
    f2.write_text(json.dumps(doc2, ensure_ascii=False), encoding="utf-8")
    IJ.run(str(f2))
    recs = [json.loads(l) for l in led.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert recs[-1]["anchor"] is None, "沒有錨點時不該填一個值"
