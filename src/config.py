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

HORIZONS = [5, 20]          # 預測期間（交易日）
UNIVERSE_SIZE = 50          # 0050 成分股數
MIN_TRAIN_ROWS = 2000       # 統計模型最低訓練樣本數

for d in (RAW, FEATURES, DOCS, CONFIG):
    d.mkdir(parents=True, exist_ok=True)
