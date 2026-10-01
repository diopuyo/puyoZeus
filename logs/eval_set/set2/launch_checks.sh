#!/bin/bash
# 最初の短区間を独立に取り直す。前セット最終戦も境界確認用に保存する。
set -eu
O=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2
available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
[ "$available" -gt 7500000 ] || exit 2
setsid -f bash "$O/job.sh" collect check 3626.0 3745.0 < /dev/null
setsid -f bash "$O/job.sh" collect s0 3412.2 3626.0 < /dev/null
