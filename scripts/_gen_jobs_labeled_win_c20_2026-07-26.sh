#!/bin/bash
# #43 c系 labeled_win 先行20本: 指標収集ジョブ生成。
# video_c{n} を $HOME/frames (ext4) へコピーし、base/gap/mid の3窓ジョブを
# _jobs_labeled_win_c20_2026-07-26.txt に書き出す (scripts/_run_safe.sh で消化する形式)。
# 窓定義は 2026-07-26 の regen ジョブと同一 (踏襲):
#   base = 0-300s / gap = 300-900s (--start-sec 300 --max-sec 600) /
#   mid  = 1200-1560s (--start-sec 1200 --max-sec 360)
#
# 使い方: bash scripts/_gen_jobs_labeled_win_c20_2026-07-26.sh <selected_videos.txt>
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1

VIDEO_LIST="${1:?selected_videos.txt (1行1 video_id) が必要}"
OUT_STUDY_DIR="data/verify/labeled_win_c20_2026-07-26/study"
mkdir -p "${OUT_STUDY_DIR}"
mkdir -p "$HOME/frames" logs

JOBS="scripts/_jobs_labeled_win_c20_2026-07-26.txt"
: > "${JOBS}"

while IFS= read -r vid; do
  [ -z "$vid" ] && continue
  n="${vid#video_c}"
  # 9p I/O ボトルネック回避のため ext4 ($HOME/frames) へコピー (未コピーの場合のみ)
  if [ ! -f "$HOME/frames/video_c${n}.mp4" ]; then
    echo "[copy] video_c${n}.mp4 -> ext4"
    cp "data/frames/video_c${n}.mp4" "$HOME/frames/"
  fi
  for suf_args in "|--max-sec 300" "_gap|--start-sec 300 --max-sec 600" "_mid|--start-sec 1200 --max-sec 360"; do
    suf="${suf_args%%|*}"
    args="${suf_args#*|}"
    echo "PYTHONPATH=. ./venv/bin/python -m scripts._collect_1t --video \$HOME/frames/video_c${n}.mp4 --out ${OUT_STUDY_DIR}/c${n}${suf}.csv ${args} > logs/labeled_win_c20_c${n}${suf}_2026-07-26.log 2>&1"
  done >> "${JOBS}"
done < "${VIDEO_LIST}"

echo "[gen] ジョブ数: $(wc -l < "${JOBS}")"
