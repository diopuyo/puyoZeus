#!/bin/bash
# 最終実装の短区間許可票を待つ。停止条件に触れた場合は全体再生しない。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2d
while [ ! -f "$OUT/LADDER.json" ]; do sleep 5; done
/usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"
exec bash scripts/run_falsedeath_v2d_shard.sh "$1"
