#!/bin/bash
# 回帰5記録が揃った時点で、採点器のテストと回帰監査を先に検収する。
set -eu
W=/mnt/d/puyo_analyzer/wt_evalset
O=$W/logs/eval_set/set2
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash "$O/test.sh"
cd "$W"
"$PY" -B -m scripts.eval_set2_labels_20261001
cd "$O/runtime"
"$PY" -B -m scripts.eval_set2_regression_20261001
date -Is > "$O/precheck.done"
