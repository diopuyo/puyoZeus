#!/bin/bash
# cycle 34 (2026-05-20): bg_fp 距離 soft prior logit shift
# cycle 33 tier 1 hard EMPTY が効かなかった反省から、 grey zone (= 距離 25-100) で
# CNN の EMPTY logit に bg_fp 距離由来のブーストを加算する soft prior 化。
# 既 default model + image_reader 改修済 (= 1st/2nd pass で bg_distance 連携)。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/review_videos/cycle34
mkdir -p logs/board_logs

VIDEOS=(
  "v89_match3_95s.mp4:v89m3"
  "v97_match11_96s.mp4:v97m11"
  "v70_match2_113s.mp4:v70m2"
)

run_viz() {
  local vid_file="$1"
  local vid_id="$2"
  local out="data/review_videos/cycle34/cycle34_${vid_id}.mp4"
  local board_log="logs/board_logs/cycle34_${vid_id}.jsonl"
  local viz_log="logs/cycle_34_viz_${vid_id}.log"
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

echo "=== cycle 34 viz @ $(date) ==="
for entry in "${VIDEOS[@]}"; do
  vid_file="${entry%%:*}"
  vid_id="${entry##*:}"
  run_viz "$vid_file" "$vid_id"
done

# evaluator 適用
mkdir -p data/verify/cycle34_eval
for entry in "${VIDEOS[@]}"; do
  vid_id="${entry##*:}"
  echo "=== evaluate cycle34 ${vid_id} ==="
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/cycle34_${vid_id}.jsonl" \
    --report-out "data/verify/cycle34_eval/cycle34_${vid_id}.json" \
    > "logs/cycle_34_eval_${vid_id}.log" 2>&1
  echo "[eval-done] cycle34 ${vid_id}"
done

echo "=== cycle 34 ALL DONE @ $(date) ===" | tee logs/cycle_34_all_done.flag
