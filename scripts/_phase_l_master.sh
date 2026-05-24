#!/bin/bash
# Phase L 本番化 一括実行 v3 (= ヘルスチェック framework 統合)
# 旧版 set -e で evaluate 段階の空 vid → 全停止 → all_done.flag 出ず
# v3 では _lib_health.sh の run_item で fail-tolerant + 中間状態保存
set +e  # run_step / run_item で個別 rc 拾うため
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
source scripts/_lib_health.sh
init_health phase_l

PHASE_L_DIR=data/phase_l
mkdir -p ${PHASE_L_DIR}/cut
mkdir -p ${PHASE_L_DIR}/seeds
mkdir -p ${PHASE_L_DIR}/review
mkdir -p data/verify/phase_l_eval
mkdir -p logs/phase_l

FFMPEG=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')

# 全動画リスト生成 (vid match_idx start dur category)
ALL_VIDEOS=$(PYTHONPATH=. ./venv/bin/python -c "
import json
data = json.load(open('data/verify/phase_l_video_selection.json'))
for s in data:
    print(f\"{s['vid']}|{s['match_idx']}|{s['start_sec']}|{s['duration']}|{s['category']}\")
")

###############################################################################
# Step 1: 試合切り出し
###############################################################################
echo "=== Step 1: 切り出し @ $(date) ==="

cut_pids=()
running=0
MAX_PARALLEL_CUT=3

while IFS='|' read -r vid match_idx start dur category; do
  [ -z "$vid" ] && continue
  key="v${vid}m${match_idx}"
  new_start=$(echo "$start - 15" | bc -l)
  new_dur=$(echo "$dur + 15" | bc -l)
  input_mp4="data/frames/video_${vid}.mp4"
  input_webm="data/frames/video_${vid}.webm"
  output="${PHASE_L_DIR}/cut/${key}_buf15s.mp4"
  log="logs/phase_l/cut_${key}.log"
  if [ -f "$input_mp4" ]; then
    "$FFMPEG" -y -ss "$new_start" -i "$input_mp4" -t "$new_dur" -c copy "$output" \
      > "$log" 2>&1 &
  elif [ -f "$input_webm" ]; then
    "$FFMPEG" -y -ss "$new_start" -i "$input_webm" -t "$new_dur" \
      -c:v libx264 -preset ultrafast -an "$output" > "$log" 2>&1 &
  else
    echo "[skip] $key (no input)"
    continue
  fi
  cut_pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL_CUT ]; then
    wait "${cut_pids[0]}"
    cut_pids=("${cut_pids[@]:1}")
    ((running--)) || true
  fi
done <<< "$ALL_VIDEOS"
wait
echo "[cut all done] @ $(date)"
ls ${PHASE_L_DIR}/cut/ | wc -l

###############################################################################
# Step 2: seed 抽出 (= 学習対象動画のみ)
###############################################################################
echo "=== Step 2: seed 抽出 @ $(date) ==="

TRAIN_VIDEOS=$(echo "$ALL_VIDEOS" | grep -E '\|(existing_train|train)$')

seed_pids=()
running=0
MAX_PARALLEL_SEED=3

while IFS='|' read -r vid match_idx start dur category; do
  [ -z "$vid" ] && continue
  key="v${vid}m${match_idx}"
  input="${PHASE_L_DIR}/cut/${key}_buf15s.mp4"
  log="logs/phase_l/seed_${key}.log"
  if [ ! -f "$input" ]; then
    echo "[seed-skip] $key (cut not found)"
    continue
  fi
  PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
    --video "$input" \
    --video-id "${key}" \
    --out-root "${PHASE_L_DIR}/seeds" \
    --max-per-color 1500 \
    --max-empty 500 \
    > "$log" 2>&1 &
  seed_pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL_SEED ]; then
    wait "${seed_pids[0]}"
    seed_pids=("${seed_pids[@]:1}")
    ((running--)) || true
  fi
done <<< "$TRAIN_VIDEOS"
wait
echo "[seed all done] @ $(date)"

###############################################################################
# Step 3: CNN 学習
###############################################################################
echo "=== Step 3: CNN 学習 @ $(date) ==="

VIDEO_IDS=$(echo "$TRAIN_VIDEOS" | awk -F'|' '{print "v" $1 "m" $2}' | tr '\n' ',' | sed 's/,$//')
echo "Training on ${VIDEO_IDS}"

PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --video-ids "${VIDEO_IDS}" \
  --store-root "${PHASE_L_DIR}/seeds" \
  --cell-arch large \
  --epochs 5 \
  --lr 1e-3 \
  --cell-save-to models/cnn_phase_l.pt \
  --augment \
  > logs/phase_l/train.log 2>&1

echo "[train done] @ $(date)"

###############################################################################
# Step 4: viz 評価 (= existing_train 8 + holdout 3 = 11 動画)
###############################################################################
echo "=== Step 4: viz @ $(date) ==="

VIZ_VIDEOS=$(echo "$ALL_VIDEOS" | grep -E '\|(existing_train|holdout)$')

viz_pids=()
running=0
MAX_PARALLEL_VIZ=2

while IFS='|' read -r vid match_idx start dur category; do
  [ -z "$vid" ] && continue
  key="v${vid}m${match_idx}"
  input="${PHASE_L_DIR}/cut/${key}_buf15s.mp4"
  output="${PHASE_L_DIR}/review/phase_l_${key}.mp4"
  board_log="logs/phase_l/viz_${key}.jsonl"
  viz_log="logs/phase_l/viz_${key}.log"
  if [ ! -f "$input" ]; then continue; fi
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" \
    --output "$output" \
    --cnn-model models/cnn_phase_l.pt \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "$viz_log" 2>&1 &
  viz_pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL_VIZ ]; then
    wait "${viz_pids[0]}"
    viz_pids=("${viz_pids[@]:1}")
    ((running--)) || true
  fi
done <<< "$VIZ_VIDEOS"
wait
echo "[viz all done] @ $(date)"

# 評価 (= 旧版で set -e + 空 vid で 全停止した真因 fix)
while IFS='|' read -r vid match_idx start dur category; do
  [ -z "$vid" ] && continue
  key="v${vid}m${match_idx}"
  board_log="logs/phase_l/viz_${key}.jsonl"
  [ ! -f "$board_log" ] && continue
  run_item eval "$key" \
    bash -c "PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
      --board-log '$board_log' \
      --report-out 'data/verify/phase_l_eval/phase_l_${key}.json' \
      > 'logs/phase_l/eval_${key}.log' 2>&1"
done <<< "$VIZ_VIDEOS"

finalize_health 0
