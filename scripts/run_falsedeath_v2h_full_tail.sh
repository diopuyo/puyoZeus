#!/bin/bash
# c群終了後は本梯子と補助起動がs群の同じ排他を待つため、空く1枠で末尾から進める。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2h
test ! -e "$OUT/full_tail.log"
exec > "$OUT/full_tail.log" 2>&1
while [ ! -e logs/eval_set/falsedeath_fix/replay/v2h/zenchi/status.json ]; do sleep 5; done
/usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"
echo "START $(date -Iseconds)"
for source in s6 s5 s4; do
  bash scripts/run_falsedeath_v2h.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2h "$source"
done
echo "END $(date -Iseconds)"
