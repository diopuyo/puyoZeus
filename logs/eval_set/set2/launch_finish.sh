#!/bin/bash
O=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2
setsid -f bash "$O/finish.sh" > "$O/finish.log" 2>&1 < /dev/null
