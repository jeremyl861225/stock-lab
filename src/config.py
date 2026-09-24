"""全域設定：路徑、常數、時區。"""
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
FEATURES = DATA / "features"
DOCS = ROOT / "docs"
CONFIG = ROOT / "config"

# append-only 記錄：系統的憲法，只准 append，不准改寫
PREDICTIONS = DATA / "predictions.jsonl"
SETTLEMENTS = DATA / "settlements.jsonl"

TPE = ZoneInfo("Asia/Taipei")
UTC = ZoneInfo("UTC")

# 5／20 日走技術與籌碼；250 日（約一年）走基本面與公司自己的擴張計畫。
# 250 日刻意不進統計模型：兩年面板在該尺度上每檔只有約 2 筆獨立觀測，
# 擬合出來的係數是雜訊。一年期只做判斷＋檢查點對帳（見 checkpoints.py）。
HORIZONS = [5, 20]          # 預測期間（交易日）－統計模型與基準線
HORIZON_1Y = 250            # 一年期：僅判斷模型，靠檢查點逐月證偽
UNIVERSE_SIZE = 50          # 0050 成分股數
MIN_TRAIN_ROWS = 2000       # 統計模型最低訓練樣本數

# 長期基本率（h 日上漲比率）—— 成績單 brier_skill 的參照氣候學。
# 來源：PIT 成分股、panel 全期間（2024-08 ~ 2026-09）每 5 交易日取樣，
# README「修掉生存者偏差之後基本率掉了多少」那張表（2026-09-19 重跑）。
# 為什麼不用「當批事後上漲比率」：任何常數預測對它的 Brier 恆為
# clim(1−clim) + (p−clim)²，技能分數必為負；單日成績單的 Brier 排序
# 因此只反映 mean_p 離當批基本率多遠（2026-09-24 審核 F8）。
# 這組數字是事前定的常數，改它要留痕（revisions.jsonl）並附新的取樣窗口。
LONGRUN_BASE_RATE = {("TW", 5): 0.536, ("TW", 20): 0.617,
                     ("US", 5): 0.529, ("US", 20): 0.553}
LONGRUN_BASE_RATE_ASOF = "20260919"

for d in (RAW, FEATURES, DOCS, CONFIG):
    d.mkdir(parents=True, exist_ok=True)
