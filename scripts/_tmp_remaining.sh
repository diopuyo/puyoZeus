#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
echo "=== 残り(rows未出力)==="
for f in logs/collect_xii_v*.log; do
  grep -q rows "$f" 2>/dev/null || echo "  $(basename "$f")"
done
echo "=== 実行中プロセスの経過時間 ==="
for pid in $(pgrep -f "scripts._collect_1t"); do
  args=$(tr '\0' ' ' < /proc/$pid/cmdline)
  out=$(echo "$args" | grep -oE 'study/[a-z0-9_]+\.csv')
  [ -z "$out" ] && continue
  et=$(ps -o etime= -p "$pid" | tr -d ' ')
  echo "  $out etime=$et"
done
