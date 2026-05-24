#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_VIDS="29 30 31 32 33"
LOG_DIR="logs"

for vid in $EXTRA_VIDS; do
  vstr=$(printf "%02d" "$vid")
  setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -u -m scripts.phase_i_collect_pseudo_labels \
    --video data/frames/video_${vstr}.mp4 \
    --video-id v${vstr} \
    --max-frames 0 \
    > $LOG_DIR/phase_i_collect_v${vstr}.log 2>&1 < /dev/null"
  echo "launched v${vstr}"
done
sleep 3
pgrep -c -f phase_i_collect_pseudo_labels
