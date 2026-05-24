#!/usr/bin/env bash
# Phase I 擬似ラベル収集 v2 (validator 改良版、Top-tier 10 動画並列)
set -euo pipefail
cd "$(dirname "$0")/.."

VIDS="29 30 31 32 33 40 51 57 70 89"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for vid in $VIDS; do
  vstr=$(printf "%02d" "$vid")
  setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -u -m scripts.phase_i_collect_pseudo_labels \
    --video data/frames/video_${vstr}.mp4 \
    --video-id v${vstr} \
    --max-frames 0 \
    > $LOG_DIR/phase_i_collect_v${vstr}.log 2>&1 < /dev/null"
  echo "launched v${vstr}"
done
sleep 3
echo "=== procs ==="
pgrep -c -f phase_i_collect_pseudo_labels
