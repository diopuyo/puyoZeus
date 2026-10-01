#!/bin/bash
# p3check の完了を待って c6 を起動する (同時6本の上限を守るため)。
L=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set
until [ -f $L/collect_p3check.done ]; do sleep 15; done
bash $L/launch_collect.sh c6
