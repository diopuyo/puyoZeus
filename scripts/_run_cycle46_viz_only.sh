#!/bin/bash
# cycle 46 viz only (= ffmpeg は完了済)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/review_videos/cycle46
mkdir -p logs/board_logs
mkdir -p data/verify/cycle46_eval

# 並列 2 で viz
viz_one() {
  local key="$1"
  local input="data/holdout_videos/${key}_buf15s.mp4"
  local output="data/review_videos/cycle46/cycle46_${key}_buf15.mp4"
  local board_log="logs/board_logs/cycle46_${key}_buf15.jsonl"
  echo "[viz-start] $key @ $(date)"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" \
    --output "$output" \
    --cnn-model models/cnn_phase_b_large_v2.pt \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle_46_viz_${key}.log" 2>&1
  echo "[viz-done] $key @ $(date)"
}
export -f viz_one

printf '%s\n' v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11 \
  | xargs -P 2 -I{} bash -c 'viz_one "{}"'

# 評価
for key in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/cycle46_${key}_buf15.jsonl" \
    --report-out "data/verify/cycle46_eval/cycle46_${key}_buf15.json" \
    > "logs/cycle_46_eval_${key}.log" 2>&1
done

echo "=== cycle 46 viz+eval DONE @ $(date) ===" | tee logs/cycle_46_done.flag
