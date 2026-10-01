#!/bin/bash
O=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2
setsid -f bash "$O/check_prefix_rows.sh" > "$O/prefix_rows.log" 2>&1 < /dev/null
