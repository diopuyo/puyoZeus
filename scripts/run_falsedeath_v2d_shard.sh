#!/bin/bash
# 全体の残りを分担する制御器。モデル・入力・採点条件は共通のまま。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2d
/usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"
case "$1" in
  set2) sources='s0 s1 s2 s3 s4 s5 s6';;
  existing) sources='review q_7gc4TgFig fcXG83vInDY mia8KCjr52g';;
  *) exit 2;;
esac
test ! -e "$OUT/shard_$1.log"
exec > "$OUT/shard_$1.log" 2>&1
echo "START $(date -Iseconds)"
for source in $sources; do
  bash scripts/run_falsedeath_v2d.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2d "$source"
done
echo "END $(date -Iseconds)"
