#!/bin/bash
W=/mnt/d/puyo_analyzer/wt_evalset
cd "$W"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
fingerprint=$(sha256sum tests/test_eval_set_score_20261001.py tests/test_eval_set_noise_20261001.py tests/test_eval_set2_score_20261001.py scripts/eval_set_score_20261001.py scripts/eval_set_noise_20261001.py scripts/eval_set_prefix_check_20261001.py scripts/eval_set2_score_20261001.py)
if [ -f logs/eval_set/set2/tests.sha ] && [ -f logs/eval_set/set2/tests.done ] && [ "$(cat logs/eval_set/set2/tests.done)" = 0 ] && [ "$(cat logs/eval_set/set2/tests.sha)" = "$fingerprint" ]; then exit 0; fi
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -B -m pytest tests/test_eval_set_score_20261001.py tests/test_eval_set_noise_20261001.py tests/test_eval_set2_score_20261001.py -q > logs/eval_set/set2/tests.log 2>&1
status=$?
echo "$status" > logs/eval_set/set2/tests.done
[ "$status" != 0 ] || printf '%s\n' "$fingerprint" > logs/eval_set/set2/tests.sha
exit "$status"
