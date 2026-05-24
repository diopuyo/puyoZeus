#!/bin/bash
# 5 動画から HSV-only 信頼判定で seed dataset を並列抽出する。
# GPU 8GB 制約のため parallel=3 (CYCLE_FINDINGS.md ルール)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p logs

MAX=500
OUT_ROOT=data/pseudo_labels_hsv_seed
PARALLEL=3

declare -a VIDEOS=(
  "v97:data/evaluation_videos/v97_match11_96s.mp4"
  "v70:data/evaluation_videos/v70_match2_113s.mp4"
  "v89m3:data/evaluation_videos/v89_match3_95s.mp4"
  "v50:data/test_unknown/v50_match1_75s_720p.mp4"
  "v91:data/test_unknown/v91_match1_75s_720p.mp4"
)

echo "[seeds] start $(date)"
for spec in "${VIDEOS[@]}"; do
  vid="${spec%%:*}"
  vpath="${spec##*:}"
  PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
    --video "$vpath" --video-id "$vid" \
    --out-root "$OUT_ROOT" \
    --max-per-color $MAX \
    --cnn-model models/cnn_phase_b_large_v3.pt \
    > "logs/extract_seed_${vid}.log" 2>&1 &
  echo "[seeds] launched $vid pid=$!"
  while [ $(jobs -rp | wc -l) -ge $PARALLEL ]; do
    sleep 5
  done
done
wait
echo "[seeds] all done $(date)"
# 集計
for vid in v97 v70 v89m3 v50 v91; do
  if [ -f "$OUT_ROOT/$vid/cell.jsonl" ]; then
    n=$(wc -l < "$OUT_ROOT/$vid/cell.jsonl")
    echo "[seeds] $vid lines=$n"
  fi
done
