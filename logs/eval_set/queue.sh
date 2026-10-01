#!/bin/bash
# ジョブ一覧 (1行 = "kind variant source") を並列数 N で順に流す。完了済み (.done が0) は飛ばす。
# 起動: setsid -f bash queue.sh <N> <jobs.txt>
E=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set
N=$1; LIST=$2
grep -v '^#' $LIST | while read -r kind variant source; do
  [ -z "$kind" ] && continue
  [ "$(cat $E/jobs/${kind}_${variant}_${source}.done 2>/dev/null)" = 0 ] && continue
  while [ "$(pgrep -fc "eval_set/job.sh")" -ge "$N" ]; do sleep 10; done
  setsid -f bash $E/job.sh $kind $variant $source < /dev/null > /dev/null 2>&1
  sleep 2
done
echo done > $E/jobs/$(basename $LIST).queue_done
