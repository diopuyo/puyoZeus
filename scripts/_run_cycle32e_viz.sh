#!/bin/bash
# cycle 32e viz 評価: cnn_cycle32e.pt で 3 動画の recognition 動画を生成
# cycle 32d と同じ 3 動画、 ただし以下フラグ付き:
#  --mask-ojama-logit (= ojama logit 推論時 mask)
#  --use-puyo-gate (= PuyoPresenceGate 配線)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

MODEL="models/cnn_cycle32e.pt"
HSV_STATE="data/per_video_hsv_ranges/_merged_default.json"
OUT_DIR="data/review_videos/cycle32e"
mkdir -p "$OUT_DIR"

VIDEOS=(
  "v89_match3_95s.mp4:v89m3"
  "v97_match11_96s.mp4:v97m11"
  "v70_match2_113s.mp4:v70m2"
)

echo "=== cycle 32e viz @ $(date) ==="
for entry in "${VIDEOS[@]}"; do
  vid_file="${entry%%:*}"
  vid_id="${entry##*:}"
  out="${OUT_DIR}/cycle32e_${vid_id}.mp4"
  log="logs/cycle_32e_viz_${vid_id}.log"
  echo "[start] ${vid_id} → ${out}"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "data/evaluation_videos/${vid_file}" \
    --output "${out}" \
    --cnn-model "${MODEL}" \
    --hsv-state "${HSV_STATE}" \
    --mask-ojama-logit \
    --use-puyo-gate \
    > "${log}" 2>&1
  echo "[done] ${vid_id} @ $(date)"
done
echo "=== ALL DONE @ $(date) ===" | tee logs/cycle_32e_viz_done.flag
