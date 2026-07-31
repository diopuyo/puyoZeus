#!/bin/bash
# labeled_win.csv 10動画 (video_29-38) 再収集ジョブ生成 (2026-07-26 夜間起動)。
#
# 目的: 2026-07-25 の認識修正4件 (d6fffe3: enable_landing_observed_color /
# enable_drift_resync_match_start_guard / enable_drift_resync_hsv_gate /
# enable_match_start_full_clear が既定 True 化) を反映した指標値で
# 既存 labeled_win.csv (data/indicators_v2/study/) と比較するための再収集。
#
# scripts/collect_indicators_v2.py:615-622 の RecognitionPipeline.load_default 呼出しは
# 上記4フラグを一切明示せずに呼んでいるため、新既定 (True) がそのまま反映される
# (scripts/visualize_advantage_overlay.py のような明示 False 上書きは存在しない)。
# よってこのジョブは追加のフラグ指定なしで新既定を再現できる。
#
# 窓定義は 2026-07-20 収集時と同一 (scripts/_jobs_collect_xii_2026-07-20.txt 踏襲):
#   base = 0-300s / gap = 300-900s (--start-sec 300 --max-sec 600) /
#   mid  = 1200-1560s (--start-sec 1200 --max-sec 360)
# --board-npz は付与しない (今回は指標CSV比較が目的、npz再生成は別タスク)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1

OUT_STUDY_DIR="data/verify/labeled_win_regen_2026-07-26/study"
mkdir -p "${OUT_STUDY_DIR}"
mkdir -p "$HOME/frames" logs

JOBS="scripts/_jobs_labeled_win_regen_2026-07-26.txt"
: > "${JOBS}"

for n in 29 30 31 32 33 34 35 36 37 38; do
  # 9p I/O ボトルネック回避のため ext4 ($HOME/frames) へコピー (未コピーの場合のみ)
  if [ ! -f "$HOME/frames/video_${n}.mp4" ]; then
    echo "[copy] video_${n}.mp4 -> ext4"
    cp "data/frames/video_${n}.mp4" "$HOME/frames/"
  fi
  for suf_args in "|--max-sec 300" "_gap|--start-sec 300 --max-sec 600" "_mid|--start-sec 1200 --max-sec 360"; do
    suf="${suf_args%%|*}"
    args="${suf_args#*|}"
    echo "PYTHONPATH=. ./venv/bin/python -m scripts._collect_1t --video \$HOME/frames/video_${n}.mp4 --out ${OUT_STUDY_DIR}/v${n}${suf}.csv ${args} > logs/labeled_win_regen_v${n}${suf}_2026-07-26.log 2>&1"
  done >> "${JOBS}"
done

echo "[gen] ジョブ数: $(wc -l < "${JOBS}")"
