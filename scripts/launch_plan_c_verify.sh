#!/usr/bin/env bash
# 完了マーカー付きの低優先度検証。共有モデルと本番設定は変更しない。
set -eu
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
py=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
out=logs/prefire_prediction/plan_c
set +e
nice -n 10 "$py" -m pytest tests/test_prefire_threat_features.py tests/test_plan_c_input_audit.py tests/test_prefire_v5c_search.py tests/test_prefire_v5b_search.py tests/test_prefire_best_play_stable.py -q > "$out/tests.log" 2>&1
code=$?
printf '%s\n' "$code" > "$out/tests.exit"
if [ "$code" -eq 0 ]; then
    nice -n 10 "$py" -m scripts.plan_c_benchmark > "$out/benchmark_native.log" 2>&1
    code=$?
    printf '%s\n' "$code" > "$out/benchmark_native.exit"
fi
exit "$code"
