#!/bin/bash
# 短区間が通過した場合だけ、固定18記録を逐次再生して採点する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2c
mkdir -p "$OUT"
worker="${1:-worker}"
test ! -e "$OUT/$worker.log"
exec > "$OUT/$worker.log" 2>&1
echo "START $(date -Iseconds)"
for source in c5 mia8KCjr52g review; do
  if [ -f "$OUT/short/$source/status.json" ]; then continue; fi
  bash scripts/run_falsedeath_v2c.sh "short_$source" -m scripts.falsedeath_v2_ladder_20261002 "$source" --mode v2c
done
bash scripts/run_falsedeath_v2c.sh ladder_score -m scripts.falsedeath_v2_ladder_20261002 score --mode v2c
# 全体側のPythonも許可票を検査する。停止時は成果物を作らない。
if ! /usr/bin/python3 -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"; then
  echo "STOP_LADDER_FAILED $(date -Iseconds)"
  exit 0
fi
echo "FULL_START $(date -Iseconds)"
for source in c1 c2 c3 c4 c5 c6 zenchi s0 s1 s2 s3 s4 s5 s6 review q_7gc4TgFig fcXG83vInDY mia8KCjr52g; do
  bash scripts/run_falsedeath_v2c.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2c "$source"
done
bash scripts/run_falsedeath_v2c.sh full_score -m scripts.falsedeath_score_20261001 --mode v2c
bash scripts/run_falsedeath_v2c.sh omissions -m scripts.falsedeath_v2c_omissions
echo "FULL_SCORED $(date -Iseconds)"
