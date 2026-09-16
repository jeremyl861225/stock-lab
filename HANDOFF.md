# HANDOFF

## 狀態：已產生第一批真實預測，含 Claude 判斷

- 基準日 **2026-09-16 收盤**。predictions.jsonl 累計 **1,600 筆**（零重複）。
- Claude 判斷：50 檔 × 5日／20日 = **100 筆**，推理鏈存於 data/reasoning.jsonl。
- 第一批成績：**5 日約 2026-09-23 可結算**，20 日約 2026-10-15。

## 今日架構變動（重要）

1. **資料源從 TWSE 改為 FinMind**。TWSE `www.twse.com.tw/rwd/` 限流極嚴：
   307（重導「因為安全性考量」頁）→ 428 → IP 冷凍。FinMind 一次請求取回整段歷史。
2. **加入除權息還原**（此前所有跨除權息日的報酬都是錯的）。
   緯穎 2026-09-02 配股後帳面「單日 -66.5%」，台股跌停僅 10%。
   - 主要來源：FinMind `TaiwanStockDividendResult`（除權前後參考價）
   - 漏網者分兩類：比例接近 1/n 視為分割並還原（國巨 2025-08-25 為 1:4）；
     比例不接近簡單分數則**截斷該日之前的歷史**，不猜因子（鴻勁 2025-04-07）
3. **預測從「方向」改為「報酬分布」**：prob_up + exp_ret + q10/q90 + 賠率比。
   只看方向會系統性誤導 —— P(up)=0.6 但漲 1%／跌 3% 的標的期望值是負的。
4. **加入四面向簡報** src/briefing.py：基本面（PER/PBR/殖利率/月營收）、
   籌碼、技術、新聞。月營收用 create_time 或推定次月 10 日做 point-in-time。
5. **模型版本 1.2.0**。舊版預測依 append-only 憲法保留，資料修正記於 data/revisions.jsonl。

## 待辦

1. **建 GitHub repo 並推上去** —— 「預測不可竄改」的最後一塊拼圖。
2. **設 LLM API key**（secret `ANTHROPIC_API_KEY` 或 `GEMINI_API_KEY`）。
   src/models/llm.py 的 prompt 已載入與人工判斷完全相同的四面向框架，
   設了 key 之後每天會自動重現，不需每天開 session。
3. **補財報資料**：FinMind `TaiwanStockFinancialStatements` 已驗證可用（毛利率／營益率／EPS），
   但 2026-09-17 抓取時免費額度用盡（每小時上限）。已排進每日流程，下一輪自動補。
   財報 point-in-time 須用法定公布期限：Q1→5/15、Q2→8/14、Q3→11/14、Q4→次年3/31。
4. **等 3 個月**再看成績單。在那之前任何結論都是回測。

## 踩過的坑（別重複）

- TWSE rwd 端點限流：307 → 428 → IP 冷凍。歷史一律走 FinMind。
- FinMind 免費版：還原股價（TaiwanStockPriceAdj）要付費，但**除權息 TaiwanStockDividendResult 免費**，
  可自行還原。每小時請求有上限，抓 300 個請求就會撞到。
- FinMind 月營收的 `create_time` 舊資料是**空字串**，直接過濾會讓 25 筆只剩 7 筆。
- 元大官網持股頁是 JS 渲染且會重導首頁；universe 改用 TWSE openapi 市值自算
  （台積電 51.45% vs 真實 0050 的 57.17%，差在流通量調整）。
- Google News 只查「台積電」前三則是員工緋聞，必須加財經關鍵字。
- 融資餘額由 0 起算的 `pct_change` 產生 inf，sklearn 直接拋錯。
- 回測取樣日期太靠近資料尾端會讓標籤尚未實現，測試要避開尾端 20 天。
