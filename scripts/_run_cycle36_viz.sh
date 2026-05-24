#!/bin/bash
# cycle 36 (2026-05-20): boost 中間値 sweep
# c34 (0.4 弱すぎ) と c35 (1.0 強すぎ) の中間 = boost 0.6 で再試行
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/review_videos/cycle36
mkdir -p logs/board_logs

VIDEOS=(
  "v89_match3_95s.mp4:v89m3"
  "v97_match11_96s.mp4:v97m11"
  "v70_match2_113s.mp4:v70m2"
)

run_viz() {
  local vid_file="$1"
  local vid_id="$2"
  local out="data/review_videos/cycle36/cycle36_${vid_id}.mp4"
  local board_log="logs/board_logs/cycle36_${vid_id}.jsonl"
  local viz_log="logs/cycle_36_viz_${vid_id}.log"
  echo "[viz-start] ${vid_id}"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "data/evaluation_videos/${vid_file}" \
    --output "${out}" \
    --cnn-model models/cnn_phase_b_large_v2.pt \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "${board_log}" \
    > "${viz_log}" 2>&1
  echo "[viz-done] ${vid_id} @ $(date)"
}

echo "=== cycle 36 viz @ $(date) ==="
for entry in "${VIDEOS[@]}"; do
  vid_file="${entry%%:*}"
  vid_id="${entry##*:}"
  run_viz "$vid_file" "$vid_id"
done

mkdir -p data/verify/cycle36_eval
for entry in "${VIDEOS[@]}"; do
  vid_id="${entry##*:}"
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/cycle36_${vid_id}.jsonl" \
    --report-out "data/verify/cycle36_eval/cycle36_${vid_id}.json" \
    > "logs/cycle_36_eval_${vid_id}.log" 2>&1
done

echo "=== cycle 36 ALL DONE @ $(date) ===" | tee logs/cycle_36_all_done.flag
