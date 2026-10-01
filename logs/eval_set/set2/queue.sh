#!/bin/bash
# 自分の計算を最大8件に制限し、空きメモリ6GB未満では新規投入しない。
set -u
O=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2
exec 9> "$O/queue.lock"
flock -n 9 || exit 4
taskfile=$O/tasks.txt
while true; do
  pending=0
  while read -r kind name variant; do
    case "$kind" in replay|regression) ;; *) exit 5 ;; esac
    tag=${kind}_${name}_${variant}
    [ -f "$O/jobs/$tag.done" ] && continue
    pending=$((pending+1))
    [ -f "$O/jobs/$tag.started" ] && continue
    active=$(pgrep -fc 'python .*scripts.(collect_eval_set_20261001|replay_exchange_event_20260926)')
    available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    host_available=$(cat "$O/host_available_kib.txt")
    ready=1
    if [ "$kind" = replay ]; then
      [ -f "$O/jobs/collect_$name.done" ] || ready=0
      if [ "$ready" = 1 ] && [ "$(cat "$O/jobs/collect_$name.done")" != 0 ]; then exit 3; fi
    fi
    [ "$active" -lt 8 ] && [ "$available" -gt 6000000 ] && [ "$host_available" -gt 5500000 ] && [ "$ready" = 1 ] || continue
    date -Is > "$O/jobs/$tag.started"
    setsid -f bash "$O/job.sh" "$kind" "$name" "$variant" < /dev/null
    echo "$(date -Is) $tag available_kb=$available active=$active"
    sleep 3
  done < "$taskfile"
  [ "$pending" -eq 0 ] && break
  sleep 20
done
echo completed
