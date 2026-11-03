#!/bin/bash
# 本番側が完了して空いた実行枠だけを、候補の後半記録へ引き継ぐ。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
while true; do
  complete=true
  for name in review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi c1 c2 c3 c4 c5 c6 s0 s1 s2 s3 s4 s5 s6; do
    if [ ! -f "logs/a1_completion/replay/prod/$name/status.json" ]; then complete=false; fi
  done
  if "$complete"; then break; fi
  sleep 30
done
export A1_MODES=a1 A1_NAMES='s6 s5 s4' A1_LOG_PREFIX=after_prod_
bash scripts/run_a1_completion.sh
