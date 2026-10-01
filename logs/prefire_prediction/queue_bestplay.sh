#!/bin/bash
# Phase 3 の再生ジョブを同時 MAX 本までで順に起動する。使い方: queue_bestplay.sh <MAX> <jobfile>
# jobfile の各行: <source> <latency> <tag> <evaluator>
cd /mnt/d/puyo_analyzer/wt_prefire
MAX=$1
while read -r src lat tag ev; do
  [ -z "$src" ] && continue
  if [ -f logs/prefire_prediction/replay/bestplay${tag}/${src}/DONE.json ]; then continue; fi
  while [ "$(pgrep -fc 'scripts.run_prefire_replay_20260930 --variant bestplay')" -ge "$((MAX*2))" ]; do sleep 20; done
  bash logs/prefire_prediction/launch_bestplay.sh "$src" "$lat" "$tag" "$ev"
  sleep 5
done < "$2"
