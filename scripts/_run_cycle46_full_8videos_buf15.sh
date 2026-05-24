#!/bin/bash
# cycle 46: 全 8 動画 (= evaluation_videos) を 15 秒バッファ付き再切り出し + 評価
# user 承認済 (= 2026-05-20、 試合切り出し境界 detector の精度問題確定 後)
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/holdout_videos
mkdir -p data/review_videos/cycle46
mkdir -p logs/board_logs
mkdir -p data/verify/cycle46_eval

FFMPEG=$(./venv/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')

# (video_id, match_start, duration) ... バッファ付き
# new_start = match_start - 15、 new_duration = (end - new_start) = original_dur + 15
# video format: v97 is .webm、 他は .mp4
declare -A VIDEOS=(
  ["v29m2"]="29 190 143 mp4"
  ["v40m7"]="40 424 122 mp4"
  ["v51m2"]="51 137 112 mp4"
  ["v57m2"]="57 190 115 mp4"
  ["v70m2"]="70 129 128 mp4"
  ["v89m3"]="89 719 84 mp4"
  ["v95m15"]="95 1015 114 mp4"
  ["v97m11"]="97 1898 111 webm"
)

# Step 1: ffmpeg 切り出し (並列 3)
echo "=== Step 1: ffmpeg 切り出し @ $(date) ==="
cut_one() {
  local key="$1"
  local info="${VIDEOS[$key]}"
  local vid=$(echo $info | cut -d' ' -f1)
  local start=$(echo $info | cut -d' ' -f2)
  local dur=$(echo $info | cut -d' ' -f3)
  local ext=$(echo $info | cut -d' ' -f4)
  local input="data/frames/video_${vid}.${ext}"
  local output="data/holdout_videos/${key}_buf15s.mp4"
  if [ "$ext" = "webm" ]; then
    "$FFMPEG" -y -ss $start -i "$input" -t $dur \
      -c:v libx264 -preset ultrafast -an "$output" \
      > "logs/cycle_46_cut_${key}.log" 2>&1
  else
    "$FFMPEG" -y -ss $start -i "$input" -t $dur -c copy "$output" \
      > "logs/cycle_46_cut_${key}.log" 2>&1
  fi
  echo "[cut-done] $key"
}
export -f cut_one
export FFMPEG

# bash の declare -A は subshell に伝わらないため、 個別呼び出し
for key in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
  info="${VIDEOS[$key]}"
  vid=$(echo $info | cut -d' ' -f1)
  start=$(echo $info | cut -d' ' -f2)
  dur=$(echo $info | cut -d' ' -f3)
  ext=$(echo $info | cut -d' ' -f4)
  input="data/frames/video_${vid}.${ext}"
  output="data/holdout_videos/${key}_buf15s.mp4"
  if [ "$ext" = "webm" ]; then
    "$FFMPEG" -y -ss $start -i "$input" -t $dur \
      -c:v libx264 -preset ultrafast -an "$output" \
      > "logs/cycle_46_cut_${key}.log" 2>&1 &
  else
    "$FFMPEG" -y -ss $start -i "$input" -t $dur -c copy "$output" \
      > "logs/cycle_46_cut_${key}.log" 2>&1 &
  fi
done
wait
echo "[cut all done] @ $(date)"
ls -la data/holdout_videos/*_buf15s.mp4

# Step 2: viz 並列 2 (= GPU memory 余裕)
echo "=== Step 2: viz 並列 2 @ $(date) ==="
viz_one() {
  local key="$1"
  local input="data/holdout_videos/${key}_buf15s.mp4"
  local output="data/review_videos/cycle46/cycle46_${key}_buf15.mp4"
  local board_log="logs/board_logs/cycle46_${key}_buf15.jsonl"
  echo "[viz-start] $key @ $(date)"
  PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
    --video "$input" \
    --output "$output" \
    --cnn-model models/cnn_phase_b_large_v2.pt \
    --hsv-state data/per_video_hsv_ranges/_merged_default.json \
    --dump-board-log "$board_log" \
    > "logs/cycle_46_viz_${key}.log" 2>&1
  echo "[viz-done] $key @ $(date)"
}
export -f viz_one

# 並列 2: 4 ペア × 2
echo v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11 | xargs -n 1 -P 2 -I{} bash -c 'viz_one "$@"' _ {}

echo "[viz all done] @ $(date)"

# Step 3: 評価
echo "=== Step 3: 評価 @ $(date) ==="
for key in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
  PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_recognition \
    --board-log "logs/board_logs/cycle46_${key}_buf15.jsonl" \
    --report-out "data/verify/cycle46_eval/cycle46_${key}_buf15.json" \
    > "logs/cycle_46_eval_${key}.log" 2>&1
done

echo "=== cycle 46 ALL DONE @ $(date) ===" | tee logs/cycle_46_done.flag
