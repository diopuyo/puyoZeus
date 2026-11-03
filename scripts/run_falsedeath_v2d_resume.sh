#!/bin/bash
# 起動口更新時に停止した制御器だけを再開する。完成済み原票は排他内で再利用する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2d
/usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"
exec >> "$OUT/worker.log" 2>&1
echo "RESUME_CONTROLLER $(date -Iseconds)"
for source in c1 c2 c3 c4 c5 c6 zenchi s0 s1 s2 s3 s4 s5 s6 review q_7gc4TgFig fcXG83vInDY mia8KCjr52g; do
  bash scripts/run_falsedeath_v2d.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2d "$source"
done
bash scripts/run_falsedeath_v2d.sh full_score -m scripts.falsedeath_score_20261001 --mode v2d
echo "FULL_SCORED $(date -Iseconds)"
