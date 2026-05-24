#!/bin/bash
# cycle 37 (2026-05-20): tier 1 threshold sweep (20→25)
# soft prior は OFF (= BG_DIST_BOOST_MAX=0)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/review_videos/cycle37
mkdir -p logs/board_logs

VIDEOS=(
  "v89_match3_95s.mp4:v89m3"
  "v97_match11_96s.mp4:v97m11"
  "v70_match2_113s.mp4:v70m2"
)

run_viz() {
  local vid_file="$1"
  local vid_id="$2"
  local out="data/review_videos/cycle37/cycle37_${vid_id}.mp4"
  local board_log="logs/board_logs/cycle37_${vid_id}.jsonl"
  local viz_log="logs/cycle_37_viz_${vid_id}.log"
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

echo "=== cycle 37 viz @ $(date) ==="
for entry in "${VIDEOS[@]}"; do
  vid_file="${entry%%:*}"
  vid_id="${entry##*:}"
  run_viz "$vid_file" "$vid_id"
done

mkdir -p data/verify/cycle37_eval
for entry in "${VIDEOS[@]}"; do
  vid_id="${entry##*:}"
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/cycle37_${vid_id}.jsonl" \
    --report-out "data/verify/cycle37_eval/cycle37_${vid_id}.json" \
    > "logs/cycle_37_eval_${vid_id}.log" 2>&1
done

echo "=== cycle 37 ALL DONE @ $(date) ===" | tee logs/cycle_37_all_done.flag
