#!/bin/bash
# 主担当がpgrepとtailで終了と直近進捗を確認する。
O=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2
date -Is
free -m
printf 'host_available_kib='
cat "$O/host_available_kib.txt"
printf '\n'
printf 'active_jobs='
pgrep -fc '[p]ython.*scripts\.(collect_eval_set_20261001|replay_exchange_event_20260926)'
for file in "$O"/jobs/*.log; do
  [ -f "${file%.log}.done" ] && continue
  printf '%s: ' "$(basename "$file")"
  tail -1 "$file"
done
printf 'completed_jobs='
find "$O/jobs" -name '*.done' -type f | wc -l
for source in review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi; do
  file="$O/jobs/regression_${source}_e19.done"
  [ ! -f "$file" ] || printf 'regression_%s.done=%s\n' "$source" "$(cat "$file")"
done
tail -3 "$O/queue.log"
[ ! -s "$O/finish.log" ] || tail -2 "$O/finish.log"
if [ -f "$O/finish.done" ]; then echo FINISHED; fi
