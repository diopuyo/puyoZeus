#!/bin/bash
# 本番互換5再生の終了後、空いた1枠で中央区間を処理する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2h2b
exec > "$OUT/middle_worker.log" 2>&1
/usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"
for source in s2 s3 s4 s1; do
  bash scripts/run_falsedeath_v2h2b.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2h2b "$source"
done
