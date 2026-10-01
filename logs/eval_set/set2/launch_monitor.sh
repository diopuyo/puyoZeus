#!/bin/bash
O=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2
setsid -f bash "$O/monitor.sh" > "$O/monitor.log" 2>&1 < /dev/null
