#!/bin/bash
# Step 1: 全 phase_l/cut 動画から per-video HSV JSON を抽出.
# 並列 3 で時間短縮、 既存 JSON は skip。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/per_video_hsv_ranges
mkdir -p logs/hsv_extract

MAX_PARALLEL=3
pids=()
running=0

for d in data/phase_l/cut/v*m*_buf15s.mp4; do
  key=$(basename "$d" _buf15s.mp4)
  # video_id は m を含まない vid 部分のみ抽出 (= v89m7 → v89)
  vid=$(echo "$key" | sed 's/m.*//')
  out="data/per_video_hsv_ranges/${vid}.json"
  if [ -f "$out" ]; then
    echo "[skip] $vid (already exists)"
    continue
  fi
  log="logs/hsv_extract/${key}.log"
  (
    PYTHONPATH=. ./venv/bin/python -m scripts.extract_per_video_hsv_ranges \
      --video "$d" \
      --video-id "$vid" \
      --cnn-model models/cnn_phase_b_large_v2.pt \
      --out "$out" \
      --max-frames 3000 \
      > "$log" 2>&1
    echo "[done] $vid"
  ) &
  pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL ]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
    ((running--)) || true
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "=== 抽出完了 @ $(date) ==="
ls data/per_video_hsv_ranges/v*.json | wc -l | xargs -I{} echo "総 JSON 数: {}"

# _merged_default.json 再生成
echo "=== merge_db_to_default 実行 ==="
PYTHONPATH=. ./venv/bin/python -m scripts.merge_db_to_default 2>&1 | tail -10
