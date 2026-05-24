#!/bin/bash
# cycle_24 やり直し: 9 video (= cycle_23 の 5 + 追加 4) で extract_hsv_seed_dataset → CReST 学習
#   - 5 → 9 video で多様性 1.8 倍、 v91 悪化解決 + 汎用化向上を期待
#   - extract_hsv_seed_dataset.py で動画から直接 STABLE × HSV-only 信頼 cell を抽出
#   - empty/ojama 追加 → CReST 学習 (cycle_23 と同じ args)
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG=logs/cycle24v2_pipeline.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date) ==="

# 9 video (raw video が手元にあるもののみ)
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

# --- 1. extract_hsv_seed_dataset で 9 video から seed 抽出 (並列 3) ---
echo "[stage1] extract HSV seed (9 videos, 3 parallel)"
mkdir -p "$SEED_RAW"
for spec in "${VIDEOS[@]}"; do
  vid="${spec%%:*}"
  vpath="${spec##*:}"
  PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
    --video "$vpath" --video-id "$vid" \
    --out-root "$SEED_RAW" \
    --max-per-color 2500 \
    --cnn-model models/cnn_phase_i_hsv_seed.pt \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --cnn-override-prob 0.70 \
    > "logs/cycle24v2_extract_${vid}.log" 2>&1 &
  while [ $(jobs -rp | wc -l) -ge $PARALLEL ]; do
    sleep 5
  done
done
wait
echo "[stage1] DONE $(date)"
echo "--- seed counts (per video) ---"
for spec in "${VIDEOS[@]}"; do
  vid="${spec%%:*}"
  if [ -f "$SEED_RAW/$vid/cell.jsonl" ]; then
    echo "  $vid: $(wc -l < $SEED_RAW/$vid/cell.jsonl) lines"
  fi
done

# --- 2. empty 追加 (extract_empty_seed) 並列 3 ---
echo "[stage2] extract empty seed (9 videos, 3 parallel)"
mkdir -p "$WITH_EMPTY"
# まず seed_raw を with_empty へコピー
for spec in "${VIDEOS[@]}"; do
  vid="${spec%%:*}"
  src="$SEED_RAW/$vid/cell.jsonl"
  dst_dir="$WITH_EMPTY/$vid"
  mkdir -p "$dst_dir"
  if [ -f "$src" ]; then
    cp "$src" "$dst_dir/cell.jsonl"
  fi
done
# empty を追加 (extract_empty_seed が cell.jsonl を append する設計と想定)
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

# --- 3. CReST 学習 ---
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
