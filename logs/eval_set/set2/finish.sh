#!/bin/bash
# 全記録・再生の正常終了後にだけ採点する。結果の採否・commitは主担当が確認する。
set -eu
W=/mnt/d/puyo_analyzer/wt_evalset
O=$W/logs/eval_set/set2
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
while true; do
  ready=1
  for part in s0 s1 s2 s3 s4 s5 s6 check; do
    [ -f "$O/jobs/collect_$part.done" ] || ready=0
  done
  while read -r kind name variant; do
    [ -f "$O/jobs/${kind}_${name}_${variant}.done" ] || ready=0
  done < "$O/tasks.txt"
  [ "$ready" -eq 1 ] && break
  sleep 30
done
for file in "$O"/jobs/*.done; do
  [ "$(cat "$file")" = 0 ] || { echo "failed: $file"; exit 2; }
done
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash "$O/test.sh"
[ "$(cat "$O/tests.done")" = 0 ]
cd "$W"
"$PY" -B -m scripts.eval_set2_labels_20261001
cd "$O/runtime"
"$PY" -B -m scripts.eval_set2_regression_20261001
cd "$W"
"$PY" -B -m scripts.eval_set2_score_20261001 --labels "$O/labels.json"
"$PY" -B -m scripts.eval_set2_report_20261001
date -Is > "$O/finish.done"
