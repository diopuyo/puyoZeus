#!/usr/bin/env bash
# Phase I.b 収集進捗確認
set -u
cd "$(dirname "$0")/.."

echo "=== alive procs ==="
pgrep -cf phase_i_collect_pseudo_labels

echo ""
echo "=== JSONL line counts (cell.jsonl) ==="
total=0
for v in 29 30 31 32 33 40 51 57 70 89; do
  f="data/pseudo_labels/v${v}/cell.jsonl"
  if [ -f "$f" ]; then
    cnt=$(wc -l < "$f")
  else
    cnt=0
  fi
  total=$((total + cnt))
  printf "v%s cell: %s\n" "$v" "$cnt"
done
echo "TOTAL cell: $total"

echo ""
echo "=== other validators (one video sample: v29) ==="
for c in score next chain hidden_row; do
  f="data/pseudo_labels/v29/${c}.jsonl"
  if [ -f "$f" ]; then
    echo "v29 ${c}: $(wc -l < $f)"
  fi
done

echo ""
echo "=== latest log line (each video) ==="
for v in 29 30 31 32 33 40 51 57 70 89; do
  log="logs/phase_i_collect_v${v}.log"
  if [ -f "$log" ]; then
    last=$(tail -1 "$log")
    sz=$(stat -c%s "$log" 2>/dev/null || stat -f%z "$log")
    printf "v%s [%sB]: %s\n" "$v" "$sz" "$last"
  fi
done

echo ""
echo "=== mtime per video dir ==="
ls -la --time-style=full-iso data/pseudo_labels/ | grep ^d
