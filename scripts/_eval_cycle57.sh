#!/bin/bash
# cycle 57 評価: 4 動画 viz (= ユーザー目視) + 8 動画 critical eval.
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle57_eval

mkdir -p data/verify/cycle57_viz
mkdir -p data/verify/cycle57_eval
mkdir -p logs/cycle57_eval

CNN_MODEL="models/cnn_cycle57.pt"
if [ ! -f "$CNN_MODEL" ]; then
  echo "ERROR: $CNN_MODEL 不在、 学習未完了か失敗"
  exit 1
fi

# Phase 1: ユーザー目視 4 動画 viz
declare -A USER_VIDEOS=(
  [v89m7]="data/phase_l/cut/v89m7_buf15s.mp4"
  [v30_match11]="data/holdout_videos/v30_match11_buf15s.mp4"
  [v30_5min]="data/holdout_videos/v30_5min_90s.mp4"
  [v97_match11]="data/holdout_videos/v97_match11_buf15s.mp4"
)
for key in "${!USER_VIDEOS[@]}"; do
  input="${USER_VIDEOS[$key]}"
  output="data/verify/cycle57_viz/${key}_cycle57.mp4"
  board_log="logs/cycle57_eval/viz_${key}.jsonl"
  report="data/verify/cycle57_viz/${key}.json"
  if [ ! -f "$input" ] || [ -f "$report" ]; then continue; fi
  echo "=== [$key] viz @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "$output" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle57_eval/viz_${key}.log" 2>&1
  [ ! -f "$board_log" ] && continue
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    > "logs/cycle57_eval/eval_${key}.log" 2>&1
  echo "[done viz] $key"
done

# Phase 2: 8 動画 評価
VIDEOS=(v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11)
for key in "${VIDEOS[@]}"; do
  input="data/baseline_videos_v3/${key}_buf15s.mp4"
  board_log="logs/cycle57_eval/eval_${key}.jsonl"
  report="data/verify/cycle57_eval/${key}.json"
  if [ ! -f "$input" ] || [ -f "$report" ]; then continue; fi
  echo "=== [$key] eval @ $(date) ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" --output "data/verify/cycle57_eval/${key}.mp4" \
    --cnn-model "$CNN_MODEL" \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle57_eval/viz8_${key}.log" 2>&1
  [ ! -f "$board_log" ] && continue
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "$board_log" --report-out "$report" \
    > "logs/cycle57_eval/eval8_${key}.log" 2>&1
  echo "[done eval] $key"
done

# Phase 3: 集計
PYTHONPATH=. ./venv/bin/python scripts/_summary_cycle57.py 2>&1 | tee data/verify/cycle57_eval/_summary.log

finalize_health 0
