#!/bin/bash
O=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2
setsid -f bash "$O/precheck.sh" > "$O/precheck.log" 2>&1 < /dev/null
