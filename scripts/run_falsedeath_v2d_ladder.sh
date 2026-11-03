#!/bin/bash
# 事前登録の短区間を通過した場合だけ18記録の全体再生へ進む。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2d
mkdir -p "$OUT"
test ! -e "$OUT/worker.log"
exec > "$OUT/worker.log" 2>&1
echo "START $(date -Iseconds)"
for source in c5 mia8KCjr52g review; do
  bash scripts/run_falsedeath_v2d.sh "short_$source" -m scripts.falsedeath_v2_ladder_20261002 "$source" --mode v2d
done
bash scripts/run_falsedeath_v2d.sh ladder_score -m scripts.falsedeath_v2_ladder_20261002 score --mode v2d
if ! /usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"; then
  echo "STOP_LADDER_FAILED $(date -Iseconds)"
  exit 0
fi
echo "FULL_START $(date -Iseconds)"
for source in c1 c2 c3 c4 c5 c6 zenchi s0 s1 s2 s3 s4 s5 s6 review q_7gc4TgFig fcXG83vInDY mia8KCjr52g; do
  bash scripts/run_falsedeath_v2d.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2d "$source"
done
bash scripts/run_falsedeath_v2d.sh full_score -m scripts.falsedeath_score_20261001 --mode v2d
echo "FULL_SCORED $(date -Iseconds)"
