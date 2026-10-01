#!/bin/bash
# 本走行の再生キューを切り離して起動する。使い方: bash launch_queue.sh <N> <jobs.txt>
E=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set
setsid -f bash $E/queue.sh "$1" "$E/$2" < /dev/null > $E/jobs_queue_$2.log 2>&1
