#!/bin/bash
# cycle 49: state machine 側で 4 連結必須 gate (= アーキ案 C)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/review_videos/cycle49
mkdir -p logs/board_logs
mkdir -p data/verify/cycle49_eval

viz_one() {
  local key="$1"
  local input="data/evaluation_videos_v2/${key}_buf15s.mp4"
  local output="data/review_videos/cycle49/cycle49_${key}.mp4"
  local board_log="logs/board_logs/cycle49_${key}.jsonl"
  echo "[viz-start] $key @ $(date)"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" \
    --output "$output" \
    --cnn-model models/cnn_phase_b_large_v2.pt \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle_49_viz_${key}.log" 2>&1
  echo "[viz-done] $key @ $(date)"
}
export -f viz_one

printf '%s\n' v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11 \
  | xargs -P 2 -I{} bash -c 'viz_one "{}"'

for key in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/cycle49_${key}.jsonl" \
    --report-out "data/verify/cycle49_eval/cycle49_${key}.json" \
    > "logs/cycle_49_eval_${key}.log" 2>&1
done

echo "=== cycle 49 ALL DONE @ $(date) ===" | tee logs/cycle_49_done.flag
