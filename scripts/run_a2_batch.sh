#!/bin/bash
# 同規模の別レーンを考慮し、A2内の再生は直列で行う。
set -eu
cd /mnt/d/puyo_analyzer/wt_a2
mode="$1"
shift
for source in "$@"; do
  if [ -f "logs/a2_safety/replay/$mode/$source/status.json" ]; then continue; fi
  bash scripts/run_a2_safety.sh "${mode}_${source}" -m scripts.a2_replay "$mode" "$source"
done
