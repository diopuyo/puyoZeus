#!/bin/bash
# 両構成の最後の記録が完了してから、固定した採点器を動かす。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
OUT=$PWD/logs/a1_completion
while true; do
  complete=true
  for mode in prod a1; do
    for name in review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi c1 c2 c3 c4 c5 c6 s0 s1 s2 s3 s4 s5 s6; do
      if [ ! -f "$OUT/replay/$mode/$name/status.json" ]; then complete=false; fi
    done
  done
  if "$complete"; then break; fi
  sleep 30
done
bash scripts/run_a1_completion.sh test
bash scripts/run_a1_completion.sh cli_test
bash scripts/run_a1_completion.sh preflight
bash scripts/run_a1_completion.sh finish
