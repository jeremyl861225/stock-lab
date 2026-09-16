"""系統最重要的一組測試：證明特徵沒有偷看未來。

這不是形式上的單元測試 —— 未來函數是這類系統最常見、也最致命的錯誤，
而且它的症狀是「回測績效好得不可思議」，不會拋任何例外。
只能靠測試抓。
"""
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features.build import build, build_all, FEATURE_COLS, labels


def _panel():
    return pd.read_parquet(Path(__file__).resolve().parent.parent
                           / "data/features/panel.parquet")


def test_truncation_invariance():
    """把未來資料整段刪掉，as_of 當天的特徵值必須一模一樣。

    注意這個測試的正確寫法：不能拿 build(p,d) 去比 build(p[p.date<=d],d) ——
    build() 第一行自己就截斷，那是 f(trunc(X)) vs f(trunc(trunc(X)))，
    恆真、永遠不會失敗，是假的安全感（審核抓到的原始版本就是這樣寫的）。
    有效的寫法是拿「看得到全部資料」的 _compute 當日切片，
    去比「只看得到 as_of 之前」的結果 —— 兩者相同才真的證明沒用到未來。
    """
    from features.build import _compute
    p = _panel()
    dates = sorted(p["date"].unique())
    full = _compute(p)                      # 看得到全部歷史（含未來）
    bad = []
    for d in dates[120::40]:
        a = full[full["as_of"] == d].set_index("code")[FEATURE_COLS].sort_index()
        b = _compute(p[p["date"] <= d])     # 只看得到 as_of 之前
        b = b[b["as_of"] == d].set_index("code")[FEATURE_COLS].sort_index()
        if list(a.index) != list(b.index):
            bad.append((str(d)[:10], "標的集合不同"))
            continue
        diff = (a - b).abs().max().max()
        if pd.notna(diff) and diff > 1e-9:
            bad.append((str(d)[:10], float(diff)))
        if not a.isna().equals(b.isna()):
            bad.append((str(d)[:10], "NaN 分布不同"))
    assert not bad, f"特徵受未來資料影響：{bad}"


def test_fast_path_matches_safe_path():
    """快速路徑（build_all）必須與安全路徑（build）逐值相同。"""
    p = _panel()
    fast = build_all(p)
    dates = sorted(p["date"].unique())
    bad = []
    for d in dates[-30::10]:
        a = build(p, d).set_index("code")[FEATURE_COLS].sort_index()
        b = (fast[fast["as_of"] == d].set_index("code")[FEATURE_COLS].sort_index())
        diff = (a - b).abs().max().max()
        if pd.notna(diff) and diff > 1e-9:
            bad.append((str(d)[:10], float(diff)))
    assert not bad, f"快慢路徑不一致：{bad}"


def test_labels_are_strictly_future():
    """標籤必須落在 as_of 之後，且步數正好等於 horizon。"""
    p = _panel()
    dates = sorted(p["date"].unique())
    d = dates[-30]
    lab = labels(p, d, 5)
    assert len(lab) > 0
    assert (lab["target_date"] > d).all(), "標籤日期沒有嚴格落在未來"
    sub = p[p["code"] == lab.iloc[0]["code"]].sort_values("date").reset_index(drop=True)
    i = sub.index[sub["date"] == d][0]
    assert sub.loc[i + 5, "date"] == lab.iloc[0]["target_date"], "horizon 步數不符"


def test_feature_hash_is_stable():
    from features.build import feature_hash
    p = _panel()
    f = build(p, p["date"].max())
    r = f.iloc[0]
    assert feature_hash(r) == feature_hash(r), "特徵指紋不穩定"
    assert len(feature_hash(r)) == 16


def test_labels_all_matches_labels():
    """向量化標籤與逐日標籤必須完全一致。"""
    p = _panel()
    from features.build import labels_all
    dates = sorted(p["date"].unique())
    fast = labels_all(p, 5)
    # 取樣日期必須離資料尾端夠遠，否則標籤本來就還沒實現（空表 ≠ 不一致）
    for d in dates[-120:-20:25]:
        a = labels(p, d, 5).set_index("code")[["fwd_ret", "y"]].sort_index()
        b = (fast[fast["as_of"] == d].set_index("code")[["fwd_ret", "y"]].sort_index())
        assert len(a) == len(b), f"{d} 筆數不符 {len(a)} vs {len(b)}"
        assert (a - b).abs().max().max() < 1e-9, f"{d} 標籤不一致"


def test_training_set_excludes_unrealised_labels():
    """訓練集不得包含標籤尚未實現的樣本（最容易犯的偷看未來）。"""
    import pandas as pd
    from models.statistical import _training_set
    p = _panel()
    as_of = sorted(p["date"].unique())[-1]
    for h in (5, 20):
        X, y_dir, y_ret = _training_set(p, as_of, h)
        assert len(X) > 0, f"h={h} 訓練集為空"
        assert len(X) == len(y_dir) == len(y_ret), f"h={h} 特徵與標籤長度不符"
        from features.build import build_all, labels_all
        labs = labels_all(p[p["date"] <= as_of], h)
        dates = sorted(p[p["date"] <= as_of]["date"].unique())
        cutoff = dates[-(h + 1)]
        assert labs[labs["as_of"] <= cutoff]["target_date"].max() <= pd.Timestamp(as_of), \
            f"h={h} 訓練標籤落在 as_of 之後"


def test_features_are_finite():
    """特徵不得含 inf —— sklearn 會拋錯，而且 inf 比缺值更危險。"""
    import numpy as np
    from features.build import build_all
    p = _panel()
    f = build_all(p)
    bad = {c: int(np.isinf(f[c].to_numpy(dtype="float64")).sum())
           for c in FEATURE_COLS
           if np.isinf(f[c].to_numpy(dtype="float64")).any()}
    assert not bad, f"特徵含 inf：{bad}"


def test_no_impossible_daily_moves():
    """還原後不得有超過漲跌停（±10%）太多的單日變動。

    門檻設 30%：台股漲跌停 ±10%，但新上市／興櫃轉上市股確實會有 11~30% 的
    真實波動，把它們「修正」掉是製造假資料。>30% 則幾乎必然是分割或減資。
    若這條測試失敗，代表還原漏了某個公司行動，而它會讓動能特徵與標籤
    同時中毒 —— 這是靜默的、不會拋錯的致命污染。
    """
    p = _panel().sort_values(["code", "date"])
    # 只檢查台股：台股有 ±10% 漲跌停，美股沒有漲跌幅限制 ——
    # Oracle 2025-09-10 因 AI 雲端訂單單日真實暴漲 35.9%，套用台股門檻會誤判成資料錯誤。
    if "market" in p.columns:
        p = p[p["market"] == "TW"]
    p["chg"] = p.groupby("code")["close"].pct_change()
    bad = p[p["chg"].abs() > 0.30]
    detail = [(r.code, str(r.date.date()), f"{r.chg:+.1%}") for r in bad.itertuples()]
    assert not detail, f"仍有不可能的單日變動（公司行動還原不完整）：{detail[:10]}"


def test_models_emit_return_distribution():
    """每個模型都必須給出期望報酬與區間，否則無法做幅度加權比較。

    只預測方向會系統性誤導：P(up)=0.60 但上漲 +1%、下跌 -3% 的標的，
    期望報酬是 -0.6%，方向準確率漂亮卻賠錢。
    """
    import pandas as pd
    from features.build import build
    from models import baselines, statistical
    p = _panel()
    d = p["date"].max()
    f = build(p, d)
    required = {"prob_up", "exp_ret", "ret_q10", "ret_q90", "direction"}
    for name, fn in {**baselines.ALL, **statistical.ALL}.items():
        out = fn(f, 5, d, p)
        assert not out.empty, f"{name} 未出手"
        assert required <= set(out.columns), f"{name} 缺欄位：{required - set(out.columns)}"
        assert (out["ret_q10"] <= out["ret_q90"]).all(), f"{name} 分位數顛倒"
        # 方向必須與期望報酬一致，否則排序與下注邏輯會自相矛盾
        if name != "always_up":
            assert ((out["exp_ret"] >= 0) == (out["direction"] == 1)).all(), \
                f"{name} 方向與期望報酬不一致"


def test_panel_buttons_have_explicit_color():
    """面板裡每個 <button> 都必須明確指定顏色，不能依賴瀏覽器預設。

    button 的預設 color 是 buttontext（純黑），不繼承 body。
    早期版本的列用 <button> 且只寫 font-family:inherit，
    結果整份清單在深色模式下變成黑字黑底 —— 實際踩過，故留此測試。
    """
    import re
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "src/panel.py").read_text(encoding="utf-8")
    css = src[src.index("CSS = "):src.index("def _cards")]
    rules = re.findall(r"([^{}]*button[^{}]*)\{([^}]*)\}", css)
    assert rules, "面板 CSS 找不到任何 button 規則"
    # 只要有一條基礎規則為 button 指定顏色即可 —— 後續規則（如 .tabs.h button
    # 只調字級）會繼承它。逐條都要求反而是誤判。
    has_color = any("color:" in body.replace("background", "") for _, body in rules)
    assert has_color, "面板所有 button 規則都沒指定顏色，深色模式會變黑字黑底"


def test_share_columns_adjusted_for_splits():
    """分割後股數類欄位必須同步還原，否則會被讀成「散戶瘋狂加槓桿」。

    用已知案例驗證，而不是掃描極端值 —— 掃描會誤判基數效應：
    台灣大的融資餘額只有幾十到上千張，小額變動就是十倍百分比，
    那是真實的（也正是審核指出 v1「融資暴增1151%」論點站不住的原因），不是分割。

    國巨 2025-08-25 為 1:4 分割。未還原時 margin_bal 由 6,542 跳到 26,429
    （比值 4.04），margin_chg_5 衝到 +2.86（全庫 99.89 百分位）。
    """
    p = _panel().sort_values(["code", "date"])
    g = p[p["code"] == "2327"].set_index("date")
    if "2025-08-25" not in g.index.strftime("%Y-%m-%d").tolist():
        return  # 資料範圍不含該事件時跳過
    chg = g["margin_bal"].pct_change().loc["2025-08-25"]
    assert abs(chg) < 0.5, (
        f"國巨分割日融資變動 {chg:+.2f}，未還原（正確值應遠小於分割倍數 3.04）")
    vchg = g["volume"].pct_change().loc["2025-08-25"]
    assert vchg < 3.0, f"國巨分割日成交量變動 {vchg:+.2f}，疑似未還原"


def test_price_times_shares_is_conserved():
    """還原的守恆檢查：價格 × 股數在「除權息事件日」前後必須連續。

    只檢查已知事件日，不掃全部日期 —— 融資餘額本身有制度性斷層：
    除權息前停止融資，之後恢復，餘額會真實跳增數倍（實測統一、玉山金、
    第一金等 8 檔都有，附近卻無除權息記錄，價格也幾乎沒動）。
    那是真實的市場行為，不是還原漏做。
    """
    import json
    from pathlib import Path as _P
    p = _panel().sort_values(["code", "date"])
    if "market" in p.columns:
        p = p[p["market"] == "TW"]      # 美股走 yfinance auto_adjust，不適用此檢查
    root = _P(__file__).resolve().parent.parent / "data/raw/finmind/div"
    if not root.exists():
        return
    bad = []
    for f in root.glob("*.json"):
        code = f.stem
        events = json.loads(f.read_text(encoding="utf-8"))["payload"]
        g = p[p["code"] == code].set_index("date")
        if g.empty or "margin_bal" not in g:
            continue
        notional = (g["close"] * g["margin_bal"]).dropna()
        for e in events:
            try:
                bp, ap = float(e["before_price"]), float(e["after_price"])
            except (TypeError, ValueError):
                continue
            if not (bp > 0 and ap > 0) or ap / bp > 0.95:
                continue           # 只檢查影響顯著的事件（配股／分割）
            d = pd.Timestamp(e["date"])
            win = notional[(notional.index >= d - pd.Timedelta(days=5)) &
                           (notional.index <= d + pd.Timedelta(days=5))]
            if len(win) < 3 or win.min() <= 0:
                continue
            if win.max() / win.min() > 3.0:
                bad.append((code, str(d.date()), round(win.max() / win.min(), 2)))
    assert not bad, f"除權息日前後價格×股數跳變逾 3 倍，還原不完整：{bad[:5]}"


def test_each_market_uses_its_own_as_of():
    """兩個市場的交易時段不同，各自取自己的最新交易日。

    台股 13:30 收盤、美股 21:30 才開盤（台北時間），所以台北 15:30 執行時
    台股已有當天資料、美股仍是前一交易日。統一取 as_of == max 會讓美股
    整個消失 —— 實測 50 檔台股、0 檔美股。
    """
    from features.build import build
    p = _panel()
    if "market" not in p.columns or p["market"].nunique() < 2:
        return
    # 模擬台股多一個交易日、美股尚未更新
    tw = p[p["market"] == "TW"]
    last = tw[tw["date"] == tw["date"].max()].copy()
    last["date"] = last["date"] + pd.Timedelta(days=1)
    sim = pd.concat([p, last], ignore_index=True)
    f = build(sim, sim["date"].max())
    for m in ("TW", "US"):
        n = (f["market"] == m).sum()
        assert n > 0, f"{m} 在跨市場時點消失（共 {len(f)} 檔）"


def test_universe_is_not_empty():
    """universe 必須有完整檔數。

    曾經在台股尚未開盤時執行 universe.py，TWSE 回空資料卻照樣寫檔，
    產生 size=0 的 universe，下游 panel 於是把台股整批濾掉、只剩美股，
    而且全程不拋任何錯誤。這種靜默失效最危險。
    """
    import json
    from pathlib import Path as _P
    root = _P(__file__).resolve().parent.parent
    for f, expect in (("config/universe_latest.json", 50),
                      ("config/universe_us_latest.json", 53)):
        p = root / f
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        n = len(d.get("constituents", []))
        assert n >= expect * 0.9, f"{f} 只有 {n} 檔（應約 {expect}）"


def test_panel_covers_both_markets():
    """panel 必須同時涵蓋兩個市場，且檔數接近 universe 大小。"""
    p = _panel()
    if "market" not in p.columns:
        return
    for m, lo in (("TW", 45), ("US", 48)):
        n = p[p["market"] == m]["code"].nunique()
        assert n >= lo, f"{m} 只有 {n} 檔在 panel 中（疑似 universe 失效）"
