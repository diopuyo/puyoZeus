#!/bin/bash
# 梯子通過後だけ逆順の記録を処理する。記録単位のflockで重複再生を防ぐ。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2e
while [ ! -f "$OUT/LADDER.json" ]; do sleep 5; done
if ! /usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"; then
  exit 0
fi
for source in mia8KCjr52g fcXG83vInDY q_7gc4TgFig review s6 s5 s4 s3 s2 s1 s0 zenchi c6 c5 c4 c3 c2 c1; do
  bash scripts/run_falsedeath_v2e.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2e "$source"
done
