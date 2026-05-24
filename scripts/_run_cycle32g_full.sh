#!/bin/bash
# cycle 32g (2026-05-19): EMPTY 採取条件拡張 + 円形マスク
#  - EMPTY_S_MAX 90→70、 EMPTY_V_MAX 170→200、 EMPTY_BG_FP_MAX 35→90
#  - circle mask (USE_CIRCLE_MASK=True、 CIRCLE_RADIUS_RATIO=0.45)
#
# 順序:
#  1. 8 動画 seed 再抽出 (= EMPTY 拡張で多様な背景採取、 並列 3、 ~30 分)
#  2. CNN 学習 (= 円形マスク有効、 ~1 分)
#  3. viz 動画生成 (= 円形マスク + ojama mask + puyo gate、 3 動画直列 ~35 分)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

###############################################################################
# Step 1: seed 再抽出 (並列 3)
###############################################################################
# 既存 cycle 32e seed を backup
for vid in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
  src="data/pseudo_labels_hsv_seed/${vid}/cell.jsonl"
  dst="data/pseudo_labels_hsv_seed/${vid}/cell_32e.jsonl"
  if [ -f "$src" ] && [ ! -f "$dst" ]; then
    cp "$src" "$dst"
  fi
done

run_seed() {
  local vid="$1"
  local video_id="$2"
  local log="logs/cycle_32g_seed_${video_id}.log"
  echo "[seed-start] ${video_id}"
  rm -f "data/pseudo_labels_hsv_seed/${video_id}/cell.jsonl"
  PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
    --video "data/evaluation_videos/${1}" \
    --video-id "${video_id}" \
    --max-per-color 1500 \
    --max-empty 500 \
    > "${log}" 2>&1
  echo "[seed-done] ${video_id}"
}
export -f run_seed

cat > /tmp/cycle32g_videos.txt <<EOF
v29_match2_156s.mp4 v29m2
v40_match7_125s.mp4 v40m7
v51_match2_97s.mp4 v51m2
v57_match2_100s.mp4 v57m2
v70_match2_113s.mp4 v70m2
v89_match3_95s.mp4 v89m3
v95_match15_99s.mp4 v95m15
v97_match11_96s.mp4 v97m11
EOF

echo "=== cycle 32g SEED EXTRACT @ $(date) ==="
cat /tmp/cycle32g_videos.txt | xargs -L 1 -P 3 bash -c 'run_seed "$0" "$1"'
echo "=== SEED DONE @ $(date) ==="

###############################################################################
# Step 2: CNN 学習
###############################################################################
echo "=== cycle 32g TRAIN @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --video-ids "v29m2,v40m7,v51m2,v57m2,v70m2,v89m3,v95m15,v97m11" \
  --store-root data/pseudo_labels_hsv_seed \
  --apply-review-filter \
  --cell-arch large \
  --epochs 5 \
  --lr 1e-3 \
  --cell-save-to models/cnn_cycle32g.pt \
  --augment \
  --use-circle-mask \
  > logs/cycle_32g_train.log 2>&1
echo "=== TRAIN DONE @ $(date) ==="

###############################################################################
# Step 3: viz 動画生成 (3 動画直列)
###############################################################################
mkdir -p data/review_videos/cycle32g

run_viz() {
  local vid_file="$1"
  local vid_id="$2"
  local out="data/review_videos/cycle32g/cycle32g_${vid_id}.mp4"
  local log="logs/cycle_32g_viz_${vid_id}.log"
  echo "[viz-start] ${vid_id}"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "data/evaluation_videos/${vid_file}" \
    --output "${out}" \
    --cnn-model models/cnn_cycle32g.pt \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --mask-ojama-logit \
    --use-puyo-gate \
    --use-circle-mask \
    > "${log}" 2>&1
  echo "[viz-done] ${vid_id} @ $(date)"
}

echo "=== cycle 32g VIZ @ $(date) ==="
run_viz "v89_match3_95s.mp4" "v89m3"
run_viz "v97_match11_96s.mp4" "v97m11"
run_viz "v70_match2_113s.mp4" "v70m2"
echo "=== VIZ DONE @ $(date) ==="

echo "=== cycle 32g ALL DONE @ $(date) ===" | tee logs/cycle_32g_all_done.flag
