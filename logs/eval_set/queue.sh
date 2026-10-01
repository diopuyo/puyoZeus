#!/bin/bash
# ジョブ一覧 (1行 = "kind variant source") を流す。同時数は 収集+再生 の合計で N 本まで。
# 入力記録 (collect/records/<source>.jsonl.gz) が揃っていないジョブは後回しにし、全件終わるまで巡回する。
# 完了済み (.done が0) と起動済み (.log がある) は飛ばす。起動: setsid -f bash queue.sh <N> <jobs.txt>
E=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set
N=$1; LIST=$2
running() { echo $(( $(pgrep -fc "eval_set/job.sh") + $(pgrep -fc "python.*collect_eval_set_20261001") )); }
while true; do
  pending=0
  while read -r kind variant source; do
    kind=${kind%$'\r'}; variant=${variant%$'\r'}; source=${source%$'\r'}
    [ -z "$kind" ] && continue
    case $kind in \#*) continue;; esac
    tag=${kind}_${variant}_${source}
    [ -f $E/jobs/$tag.log ] && continue
    pending=1
    [ "$source" != zenchi ] && [ ! -f $E/collect/records/$source.jsonl.gz ] && continue
    while [ "$(running)" -ge "$N" ]; do sleep 10; done
    mkdir -p $E/jobs; : > $E/jobs/$tag.log
    setsid -f bash $E/job.sh $kind $variant $source < /dev/null > /dev/null 2>&1
    sleep 2
  done < $LIST
  [ $pending = 0 ] && break
  sleep 30
done
echo done > $E/jobs/$(basename $LIST).queue_done
