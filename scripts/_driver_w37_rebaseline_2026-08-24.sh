#!/bin/bash
# W36 + W37 是正後の 8 動画 再基準化 (2026-08-24、user 判断1=案A)。
#
#   W36: --gravity-settle-state の配線漏れを是正 (本番と同じ構成で測る)
#   W37: 試合外 (盤面が画面に無い) 区間を除いた値を per_video.offmatch に併記
#
# 主指標 acc は未除外のまま残すので、新旧を同じ JSON 内で併記できる
# (feedback_paired_comparison_fixed_population_2026-08-20)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
PY="./venv/bin/python"
OUT=data/verify/w37_rebaseline_2026-08-24
mkdir -p "$OUT"
ARGS="--videos v29,v40,v51,v57,v70,v89,v95,v97 --holdout v89,v97 --workers 4 --game-event-chain-exit"

PYTHONPATH=. nice -n 5 $PY scripts/measure_stable_cell_acc.py $ARGS \
  --output "$OUT/off.json" > "$OUT/off.log" 2>&1
echo done_off >> "$OUT/progress.log"

PYTHONPATH=. nice -n 5 $PY scripts/measure_stable_cell_acc.py $ARGS \
  --enable-formula-freeze-fix --output "$OUT/on.json" > "$OUT/on.log" 2>&1
echo done_on >> "$OUT/progress.log"

echo ALL_DONE >> "$OUT/progress.log"
