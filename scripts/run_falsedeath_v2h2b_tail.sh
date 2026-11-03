#!/bin/bash
# 梯子合格後だけ逆順で再生する。記録別flockで主ワーカーとの二重再生を防ぐ。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2h2b
exec > "$OUT/tail_worker.log" 2>&1
while [ ! -f "$OUT/LADDER.json" ]; do sleep 5; done
/usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"
for source in mia8KCjr52g fcXG83vInDY q_7gc4TgFig review s6 s5 s4 s3 s2 s1 s0 zenchi c6 c5 c4 c3 c2 c1; do
  bash scripts/run_falsedeath_v2h2b.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2h2b "$source"
done
