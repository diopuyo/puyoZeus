#!/bin/bash
# cycle 32d viz 評価: cnn_cycle32d.pt で 3 動画の recognition 動画を生成
# 観点:
#  - v89m3: 1P 青背景の前科 → 背景誤認チェック
#  - v97m11: puyo→empty 副作用前科 → fail-silent チェック
#  - v70m2: red 3 件のみ採取 → red 認識力チェック
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

MODEL="models/cnn_cycle32d.pt"
HSV_STATE="data/per_video_hsv_ranges/_merged_default.json"
OUT_DIR="data/review_videos/cycle32d"
mkdir -p "$OUT_DIR"

VIDEOS=(
  "v89_match3_95s.mp4:v89m3"
  "v97_match11_96s.mp4:v97m11"
  "v70_match2_113s.mp4:v70m2"
)

echo "=== cycle 32d viz @ $(date) ==="
for entry in "${VIDEOS[@]}"; do
  vid_file="${entry%%:*}"
  vid_id="${entry##*:}"
  out="${OUT_DIR}/cycle32d_${vid_id}.mp4"
  log="logs/cycle_32d_viz_${vid_id}.log"
  echo "[start] ${vid_id} → ${out}"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "data/evaluation_videos/${vid_file}" \
    --output "${out}" \
    --cnn-model "${MODEL}" \
    --hsv-state "${HSV_STATE}" \
    > "${log}" 2>&1
  echo "[done] ${vid_id} @ $(date)"
done
echo "=== ALL DONE @ $(date) ===" | tee logs/cycle_32d_viz_done.flag
