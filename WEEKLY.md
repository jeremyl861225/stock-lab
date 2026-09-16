# 每週深度版 SOP

**開新 session 執行，不要延續舊對話。** 舊對話會被壓縮，壓縮後的摘要會丟失
「金控月營收受保費與會計認列擺佈」這種具體到能救一次判斷的細節。
這份文件加上 `LESSONS.md`、`HANDOFF.md`、`revisions.jsonl`，比任何 session 記憶可靠。

專案：`/Users/jeremy/Desktop/Claude code/stock-lab`　Python：`./.venv/bin/python`

---

## 與每日版的差異

每日版（`DAILY.md`）只做一件事：**產生今天的判斷**。
每週版做的是**改進系統本身**——每日版沒空做、也不該在盤前做的事。

| | 每日 | 每週 |
|---|---|---|
| 產生判斷 | ✓ | 順帶（若當天尚未做） |
| 讀成績單 | 掃一眼 | 逐筆分析錯在哪 |
| 派審核 agent | ✗ | ✓ 核心工作 |
| 修程式缺陷 | ✗ | ✓ |
| 更新 LESSONS.md | 發現新型態才加 | 必做 |

---

## 一、先看系統現況（10 分鐘）

```bash
cd "/Users/jeremy/Desktop/Claude code/stock-lab"
git pull --rebase
./.venv/bin/python -m pytest tests/ -q
./.venv/bin/python src/accuracy.py
./.venv/bin/python src/score.py 2>/dev/null | head -40
tail -3 data/revisions.jsonl | python3 -m json.tool 2>/dev/null | head -40
```

讀 `HANDOFF.md` 的「未修問題」清單。

## 二、錯誤歸因分析（有結算資料後，這是最重要的一節）

從 `data/settlements.jsonl` 找**錯得最離譜的 10–15 筆**（`correct=0` 且 `|actual_return|` 最大），
對每一筆回頭讀 `data/reasoning.jsonl` 裡當時的 `thesis` / `facts` / `inference` / `falsifier`，
分類成三種：

| 錯誤類型 | 判準 | 處置 |
|---|---|---|
| **事實錯了** | 當時引用的數字或事件與實際不符 | 加進 `LESSONS.md`；若是資料源問題就修管線 |
| **推論錯了** | 事實對，但從事實到結論的邏輯站不住 | 加進 `LESSONS.md` 的對應分類 |
| **推論對但市場不理會** | 事實與邏輯都對，價格就是沒反應 | **這是市場效率，不是錯誤** — 該考慮放棄這個角度 |

**分不清這三種就不會進步。** 第三類最容易被誤當成第二類，然後白白調整一個本來就對的模型。

另外檢查 `falsifier`：當初寫的否證條件有沒有被觸發？
若觸發了卻沒改變判斷，代表否證條件只是場面話。

## 三、派審核 agent（每週至少一輪）

依上週改動的範圍挑 2–3 個面向，**同時派出、彼此不重疊**。
歷史上每一輪都抓到真實錯誤，沒有一次是白跑的。三個已驗證有效的面向：

1. **判斷內容查核**：103 檔的每一條數值宣稱對照 `briefing.parquet`；
   所有「最高／最低／唯一／全場」用排序驗；漲跌幅前十檔與所有 high/medium 判斷
   逐檔讀新聞檔查歸因；MACRO 的每個數字要能重算。
2. **系統與流程**：`daily.py` 兩階段、`DAILY.md` 是否自足可執行、
   Actions 與本機流程會不會打架、PWA 在子路徑下的相對路徑、測試盲點、
   未修問題在「明天就要跑」的前提下哪一個實害最大。
3. **數值計算鏈**：briefing → 判斷檔 → JSON → predictions.jsonl → ranking → 面板顯示，
   每一步的數字是否正確傳遞；面板顯示值與重算值逐欄比對。

**prompt 必須要求「實際跑程式驗證，不要只讀程式碼」，並要求只回報驗證過的問題。**
過去三輪最嚴重的幾項（賠率比差 1.6 倍、跨市場 as_of 失聯、universe 空檔）
都是實跑才看得到、靜態閱讀看不到的。

提醒 agent：專案資料不要動，臨時腳本寫 /tmp。
另外**不要讓 agent 跑 `daily.py prepare`** —— 它會重建 universe 與 panel，
曾經因此在台股未開盤時寫出 size=0 的 universe，導致台股整批消失。

## 四、處理審核結果

1. 依「明天就會出事」的程度排序，不是依修起來容不容易
2. 每修一項就加一個測試 —— 沒有測試的修正會再犯
3. 錯誤的判斷**不刪**，用 `--revise` 寫新版：
   `./.venv/bin/python src/ingest_judgment.py --revise judgments/<新版>.json`
4. 把審核發現寫進 `data/revisions.jsonl`（含已修與未修）
5. 新的錯誤型態寫進 `LESSONS.md` 對應分類，註明日期與案例

## 五、更新 HANDOFF.md 並推送

未修問題清單要保持誠實 —— 修好的移除、新發現的加入、
並標明哪一項在「每天自動跑」的前提下實害最大。

---

## 目前的未修問題（依實害排序，2026-09-17）

1. **生存者偏差**（最痛）。不只污染評分，更經由 `stat` 模型的訓練集**污染 live 預測**：
   `mkt_ret_5/20` 與 `xs_ret_20` 是用「今天的市值前 50」算的橫斷面統計，
   而這 50 檔兩年等權 +272%、49/50 上漲 —— 等於每天餵模型一個「什麼都會漲」的先驗。
   修法：每日存 universe 快照、`universe.load(as_of)` 全面帶入、
   panel 保留所有曾入選過的代號。需要歷史各時點的全市場市值資料。
2. **`feature_hash` 對 claude 模型全空**（`ingest_judgment.py` 寫死 `""`）。
   憲法 C 對唯一無法自動化、也最有價值的那個模型完全落空。
3. **`_binom_p` 假設 i.i.d.**：Monte Carlo 實測名目 α=0.05、實際拒絕率 0.34–0.36（寬鬆 7 倍），
   問題在同日 50 檔被當成 50 個獨立樣本。`crosssec` 有做重疊調整，`score` 沒有。
   三個月後會宣稱有 edge。
4. **EV₅ = EV₂₀/4 的推論不完整**：skew 項只 ∝√h，實測比值中位數 0.375 而非 0.25。
   排序不受影響，但 docstring 的推論要更正。
5. **停牌時 horizon 失真**：`pct_change` 與 `i+horizon` 都是位置性的，
   2327 停止買賣期間「5 日預測」實際用 12 個市場日結算。兩年一例。
