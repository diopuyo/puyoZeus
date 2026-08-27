#!/bin/bash
# 収集プロセスの動画読み位置から進捗%を推定する (fdinfo pos / ファイルサイズ)
for pid in $(pgrep -f 'collect_lean_1t'); do
  vid=$(tr '\0' '\n' < /proc/$pid/cmdline 2>/dev/null | grep -A1 '^--video$' | tail -1)
  [ -z "$vid" ] && continue
  fdnum=""
  for l in /proc/$pid/fd/*; do
    tgt=$(readlink "$l")
    if [ "$tgt" = "$vid" ]; then fdnum=$(basename "$l"); break; fi
  done
  [ -z "$fdnum" ] && continue
  pos=$(awk '/^pos:/{print $2}' /proc/$pid/fdinfo/$fdnum)
  size=$(stat -c %s "$vid")
  echo "$(basename $vid) $((pos * 100 / size))%"
done
