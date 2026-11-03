#!/bin/bash
# 回帰検査と互換再生が終わった枠を全体再生へ回す。計算は本梯子と合わせ最大3本。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2h
test ! -e "$OUT/full_extra.log"
exec > "$OUT/full_extra.log" 2>&1
while ! tail -n 1 "$OUT/regression_tests.log" | grep -q 'passed'; do sleep 5; done
while [ ! -e "$OUT/compat_v2g/review/COMPATIBILITY.json" ]; do sleep 5; done
/usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"
echo "START $(date -Iseconds)"
(
  for source in s0 s1 s2 s3 s4 s5 s6; do
    bash scripts/run_falsedeath_v2h.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2h "$source"
  done
) &
first=$!
(
  for source in review q_7gc4TgFig fcXG83vInDY mia8KCjr52g; do
    bash scripts/run_falsedeath_v2h.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2h "$source"
  done
) &
second=$!
wait "$first"
wait "$second"
echo "END $(date -Iseconds)"
