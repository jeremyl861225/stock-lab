#!/bin/bash
# 續跑基本面回填。額度用盡就等 20 分鐘再試，只補缺口（靠落盤快取）。
cd "$(dirname "$0")/.."
for i in $(seq 1 12); do
  ./.venv/bin/python src/collect/fundamentals.py 2>&1 | tail -6
  left=$(./.venv/bin/python -c "
import sys; sys.path.insert(0,'src')
from collect import fundamentals as F, universe
print(sum(len(v) for v in F.missing(sorted(universe.codes_ever())).values()))" 2>/dev/null)
  echo "[$(date +%H:%M)] 第 $i 輪結束，還缺 ${left:-?} 個請求"
  [ "$left" = "0" ] && { echo "DONE 基本面長料回填完成"; exit 0; }
  sleep 1200
done
echo "STALLED 12 輪後仍未完成，還缺 ${left:-?}"
