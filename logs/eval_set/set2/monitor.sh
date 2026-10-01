#!/bin/bash
# 20秒間隔の資源記録。4GBを割る前に新規投入を止める監視資料。
O=/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/set2
while true; do
  available=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
  active=$(pgrep -fc 'python .*scripts.(collect_eval_set_20261001|replay_exchange_event_20260926)')
  printf '%s\t%s\t%s\n' "$(date -Is)" "$available" "$active" >> "$O/resources.tsv"
  [ "$active" -eq 0 ] && [ -f "$O/RESULT.json" ] && break
  sleep 20
done
