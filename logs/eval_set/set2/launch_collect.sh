#!/bin/bash
# 試合境界で6分割、各30秒助走は元収集コマンドから維持。
set -eu
W=/mnt/d/puyo_analyzer/wt_evalset
O=$W/logs/eval_set/set2
setsid -f bash "$O/job.sh" collect s1 3626.0 4293.2 < /dev/null
setsid -f bash "$O/job.sh" collect s2 4293.2 4924.033 < /dev/null
setsid -f bash "$O/job.sh" collect s3 4924.033 5482.066 < /dev/null
setsid -f bash "$O/job.sh" collect s4 5482.066 6066.6 < /dev/null
setsid -f bash "$O/job.sh" collect s5 6066.6 6664.166 < /dev/null
setsid -f bash "$O/job.sh" collect s6 6664.166 7033.6 < /dev/null
