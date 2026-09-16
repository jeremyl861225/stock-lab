# stock-lab — 每日台股預測與驗證

每天對市值前 50 大台股產生「未來 5 日 / 20 日的報酬分布」，把預測**事前鎖死**，
到期後自動對答案，並與笨基準線並排計分。

**這個系統的目的不是賺錢，是防止自我欺騙。**
人腦有後見之明偏誤：三個月後你只會記得自己看對的那幾次。

期望值 = P(漲) × 上漲時幅度 + P(跌) × 下跌時幅度。
只看方向機率會系統性誤導 —— 一檔 P(漲)=0.6 但漲 1%／跌 3% 的標的，期望值是負的。

---

## ⚠️ 先讀這段：回測結論目前不可用

整份 2 年 panel 使用**「今天」的市值前 50**。這 50 檔兩年等權 **+272%、49/50 上漲（98%）**。

後果有兩層：
1. `always_up` 的基本率有相當部分是「因為這批股票後來漲了才被選進 universe」造出來的
2. 更嚴重：`mkt_ret_5` / `mkt_ret_20` / `xs_ret_20` 三個特徵本身就是這批倖存者的
   橫斷面統計 —— **特徵本體帶有前視資訊**

**在修好之前，下面所有「模型 vs 基準線」的比較與 RankIC 都不能當作證據。**
下面列出數字是為了可重現，不是為了下結論。

---

## 四道防線（其中一道曾經是假的）

| 防線 | 做法 |
|---|---|
| 預測不可竄改 | `predictions.jsonl` 只准 append；身分鍵為 (as_of, horizon, model, code)，**不含版本**；計分時只取每組最新一筆 |
| 必有笨基準線 | `always_up` / `random` / `momentum` 每天照常出手，永不關閉 |
| 禁止未來函數 | 特徵一律 rolling/shift；訓練集截止日 = `as_of − horizon`；12 項測試逐日驗證 |
| 顯著性要誠實 | 橫斷面 t 值依樣本重疊倍數折減後才判讀 |

第三道曾經是假的：`test_truncation_invariance` 原本比較 `f(trunc(X))` 與
`f(trunc(trunc(X)))`——`build()` 第一行自己就截斷，所以它**數學上不可能失敗**，
卻被寫進 README 當成第一道防線。已改為拿「看得到全部資料」的 `_compute` 當日切片
去比「只看得到過去」的結果。

---

## 回測數字（2025-09 ~ 2026-09，50 個取樣日，可重現）

**一、方向準確率：所有模型都沒贏過無腦猜漲**

| 期間 | always_up | stat_logit | stat_gbdt | momentum | random |
|---|---|---|---|---|---|
| 5 日 | **58.6%** | 55.4% | 56.6% | 56.4% | 51.0% |
| 20 日 | **68.4%** | 64.4% | 62.0% | 68.4% | 50.0% |

**二、橫斷面選股能力：兩個指標給出相反結論**

| 期間 | 模型 | RankIC | IC 調整 t | 前20−後20 | spread 調整 t |
|---|---|---|---|---|---|
| 5 日 | momentum | +0.071 | **2.14** | +1.74% | 2.64 |
| 5 日 | stat_logit | +0.065 | 1.69 | +2.23% | **3.21** |
| 20 日 | stat_logit | +0.149 | 1.70 | +8.64% | **2.19** |
| 20 日 | momentum | +0.122 | **2.14** | +5.95% | 1.85 |

RankIC 衡量「整體排序的相關性」，spread 衡量「頭尾兩端的差距」——
一個模型可以中段排序很亂（IC 低）但頭尾分得很開（spread 大）。**兩者不可互相代替**，
而先前的表格只印 IC 的 t 卻放在 spread 欄旁，剛好會讓人讀反。

**三、為什麼即使 t > 2 也不能下結論**

`momentum` 兩個期間的 IC 調整 t 都是 2.14，看似通過檢驗。但這 50 檔是
「後來漲最多的股票」，**動能策略在倖存者樣本上天然佔優**——這正是生存者偏差最典型的表現。
此外 `_binom_p` 假設 i.i.d.，Monte Carlo 實測名目 α=0.05、實際拒絕率 **0.299**（寬鬆約 6 倍）。

---

## 使用

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python src/collect/universe.py   # universe（市值前 50）
./.venv/bin/python src/collect/finmind.py    # 回補價量／法人／融資券／除權息／估值／月營收
./.venv/bin/python src/features/panel.py     # 組 panel（含除權息還原）
./.venv/bin/python src/briefing.py           # 四面向簡報
./.venv/bin/python src/predict.py            # 今日預測
./.venv/bin/python src/settle.py             # 結算到期預測
./.venv/bin/python src/ranking.py            # 期望值排序
./.venv/bin/python src/panel.py              # 手機面板 → docs/index.html
./.venv/bin/python -m pytest tests/ -q       # 12 項防線測試
```

GitHub Actions 每交易日台北 18:00 自動執行並 commit，**git 歷史即不可竄改的稽核軌跡**。

Claude 判斷需每日開 session 產生，或設 `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`
讓 `src/models/llm.py` 自動跑（其 prompt 已載入與人工判斷相同的四面向框架）。

## 資料來源

| 來源 | 內容 |
|---|---|
| FinMind | 價量、三大法人、融資券、**除權息參考價**、本益比、月營收 |
| TWSE OpenAPI | 上市公司股數 → 市值（用於建 universe） |
| Google News RSS | 個股與大盤財經新聞 |

**不要用 TWSE `STOCK_DAY` 抓歷史**：一次只給一檔一個月，限流極嚴 ——
307（重導「因為安全性考量」頁）→ 428 → IP 冷凍。FinMind 150 個請求、89 秒解決。

除權息必須還原，而且**價格與股數兩邊都要**：價格乘 factor、股數除以 factor。
只還原價格的話，國巨 1:4 分割會讓 `margin_chg_5` 衝到 +2.86（全庫 99.89 百分位），
被模型讀成「散戶瘋狂加槓桿」。

## 限制

1. **生存者偏差未修**（見最上方）。
2. **LLM 判斷無法回測** —— 它知道歷史結果。成績只能從上線日往後累積。
3. **研究者自由度** —— 特徵與模型是事後挑的，walk-forward 消除不了。
4. **交易成本未計** —— 即使找到 edge，扣掉手續費與證交稅後未必有利可圖。
5. 其餘已知問題見 `HANDOFF.md`，全部發現記於 `data/revisions.jsonl`。

本專案是研究與紀律工具，**不構成投資建議**。
