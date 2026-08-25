#!/bin/bash
# lockfix 再収集の完了番人: 全 collector 終了まで5分ごとに進捗記録、終了で exit
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
LOG=$ROOT/logs/_monitor_lockfix_r3_2026-08-19.log
while true; do
  n=$(pgrep -c -f 'collect_lean_1t')
  {
    echo "=== $(date +%H:%M:%S) collectors=$n"
    uptime
    free -g | sed -n 2p
    bash $ROOT/scripts/_psdiag_lockfix_progress_2026-08-19.sh 2>/dev/null
    ls $ROOT/data/indicators_v2/boards_lean_lockfix_2026-08-19/*.npz 2>/dev/null | wc -l
  } >> $LOG
  if [ "$n" -eq 0 ]; then
    echo "ALL-COLLECT-DONE $(date +%H:%M:%S)" >> $LOG
    echo ALL-COLLECT-DONE
    exit 0
  fi
  sleep 300
done
