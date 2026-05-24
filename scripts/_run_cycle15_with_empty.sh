#!/bin/bash
# cycle_15: empty seed 追加 → 再 fine-tune → 5 動画 cycle_15 viz → 集計
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

LOG=logs/cycle15_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date) ==="

OUT=data/pseudo_labels_hsv_seed_with_empty

# --- 1. 5 動画分の既存 puyo seed をコピー ---
echo "[stage1] copy 5-color seeds"
for vid in v97 v70 v89m3 v50 v91; do
  src=data/pseudo_labels_hsv_seed_no_ojama/$vid/cell.jsonl
  dst_dir=$OUT/$vid
  mkdir -p "$dst_dir"
  if [ -f "$src" ]; then
    cp "$src" "$dst_dir/cell.jsonl"
    echo "[copy] $vid: $(wc -l < $dst_dir/cell.jsonl) lines"
  fi
done

# --- 2. empty seed 5 動画並列抽出 (base model = cnn_phase_b_large_v3.pt) ---
echo "[stage2] extract empty seed (5 videos, 3 parallel)"
declare -a VIDEOS=(
  "v97:data/evaluation_videos/v97_match11_96s.mp4"
  "v70:data/evaluation_videos/v70_match2_113s.mp4"
  "v89m3:data/evaluation_videos/v89_match3_95s.mp4"
  "v50:data/test_unknown/v50_match1_75s_720p.mp4"
  "v91:data/test_unknown/v91_match1_75s_720p.mp4"
)
PARALLEL=3
for spec in "${VIDEOS[@]}"; do
  vid="${spec%%:*}"
  vpath="${spec##*:}"
  PYTHONPATH=. ./venv/bin/python -m scripts.extract_empty_seed \
    --video "$vpath" --video-id "$vid" \
    --out-root "$OUT" \
    --max-empty 500 \
    --cnn-model models/cnn_phase_b_large_v3.pt \
    > "logs/extract_empty_${vid}.log" 2>&1 &
  while [ $(jobs -rp | wc -l) -ge $PARALLEL ]; do
    sleep 5
  done
done
wait
echo "[stage2] DONE $(date)"
for vid in v97 v70 v89m3 v50 v91; do
  if [ -f "$OUT/$vid/cell.jsonl" ]; then
    echo "[merged] $vid: $(wc -l < $OUT/$vid/cell.jsonl) lines"
  fi
done

# --- 3. 再 fine-tune (empty + 5 puyo) ---
echo "[stage3] fine-tune cell_color (with empty)"
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --all \
  --store-root "$OUT" \
  --cell-base-model models/cnn_phase_b_large_v3.pt \
  --cell-save-to models/cnn_phase_i_hsv_seed_v2.pt \
  --cell-arch large \
  --class-balance \
  --augment \
  --epochs 5 \
  2>&1 | tail -100
if [ ! -f models/cnn_phase_i_hsv_seed_v2.pt ]; then
  echo "[stage3] FAILED: model not created"
  exit 1
fi
echo "[stage3] DONE $(date)"
ls -la models/cnn_phase_i_hsv_seed_v2.pt

# --- 4. cycle_15 viz ---
echo "[stage4] cycle_15 viz"
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle \
  --cycle 15 --parallel 3 \
  --cnn-model models/cnn_phase_i_hsv_seed_v2.pt \
  --cnn-override-prob 0.70 \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  2>&1 | tail -50
echo "[stage4] DONE $(date)"

# --- 5. 集計 ---
echo "[stage5] cycle metrics (cycle_5 / cycle_12 / cycle_14 / cycle_15)"
PYTHONPATH=. ./venv/bin/python -m scripts.cycle_metrics \
  'viz_v*_multicycle_5.log' \
  'viz_v*_multicycle_12.log' \
  'viz_v*_multicycle_14.log' \
  'viz_v*_multicycle_15.log' \
  > logs/cycle_15_metrics.json
echo "[stage5] DONE $(date)"
echo "=== END $(date) ==="
