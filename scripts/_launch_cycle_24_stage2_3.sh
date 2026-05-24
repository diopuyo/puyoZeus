#!/bin/bash
# cycle_24 stage 2 (empty 追加) + stage 3 (CReST 学習) 個別起動
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG=logs/cycle24v2_stage2_3.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date) ==="

declare -a VIDEOS=(
  "v97:data/evaluation_videos/v97_match11_96s.mp4"
  "v70:data/evaluation_videos/v70_match2_113s.mp4"
  "v89m3:data/evaluation_videos/v89_match3_95s.mp4"
  "v50:data/test_unknown/v50_match1_75s_720p.mp4"
  "v91:data/test_unknown/v91_match1_75s_720p.mp4"
  "v29m2:data/evaluation_videos/v29_match2_156s.mp4"
  "v40m7:data/evaluation_videos/v40_match7_125s.mp4"
  "v51m2:data/evaluation_videos/v51_match2_97s.mp4"
  "v57m2:data/evaluation_videos/v57_match2_100s.mp4"
)

SEED_RAW=data/pseudo_labels_hsv_seed_all9
WITH_EMPTY=data/pseudo_labels_hsv_seed_with_empty_all9
PARALLEL=3

# --- stage 2: empty 追加 ---
echo "[stage2] copy seed_raw to with_empty + extract empty (9 videos, 3 parallel)"
mkdir -p "$WITH_EMPTY"
for spec in "${VIDEOS[@]}"; do
  vid="${spec%%:*}"
  src="$SEED_RAW/$vid/cell.jsonl"
  dst_dir="$WITH_EMPTY/$vid"
  mkdir -p "$dst_dir"
  if [ -f "$src" ]; then
    cp "$src" "$dst_dir/cell.jsonl"
  fi
done
for spec in "${VIDEOS[@]}"; do
  vid="${spec%%:*}"
  vpath="${spec##*:}"
  PYTHONPATH=. ./venv/bin/python -m scripts.extract_empty_seed \
    --video "$vpath" --video-id "$vid" \
    --out-root "$WITH_EMPTY" \
    --max-empty 500 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    > "logs/cycle24v2_empty_${vid}.log" 2>&1 &
  while [ $(jobs -rp | wc -l) -ge $PARALLEL ]; do
    sleep 5
  done
done
wait
echo "[stage2] DONE $(date)"
echo "--- with_empty counts ---"
for spec in "${VIDEOS[@]}"; do
  vid="${spec%%:*}"
  if [ -f "$WITH_EMPTY/$vid/cell.jsonl" ]; then
    echo "  $vid: $(wc -l < $WITH_EMPTY/$vid/cell.jsonl) lines"
  fi
done

# --- stage 3: CReST 学習 ---
echo "[stage3] CReST fine-tune (epochs=10, augment, oversample-alpha=0.5)"
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color --all \
  --store-root "$WITH_EMPTY" \
  --cell-base-model models/cnn_phase_i_hsv_seed.pt \
  --cell-save-to models/cnn_phase_b_crest_v2.pt \
  --cell-arch large \
  --epochs 10 --augment \
  --oversample-alpha 0.5 \
  --focal-gamma 2.0 \
  --logit-adjust-tau 1.0
if [ ! -f models/cnn_phase_b_crest_v2.pt ]; then
  echo "[stage3] FAILED: model not created"
  exit 1
fi
ls -la models/cnn_phase_b_crest_v2.pt
echo "[stage3] DONE $(date)"
echo "=== END $(date) ==="
