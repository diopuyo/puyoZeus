#!/usr/bin/env bash
# B18cと同じ保存入力・動画区間を、B19のコードで再測定する。
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/rt/puyo_analyzer
mkdir -p logs/live_b19
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PYTHON=${PUYO_TEST_PYTHON:-/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python}
exec > logs/live_b19/verification.log 2>&1 < /dev/null
"$PYTHON" -m scripts.verify_live_b19 --copy-cost
for source in review zenchi; do
  nice -n 19 "$PYTHON" -u -m scripts.verify_live_b18c --source "$source" --output-root logs/live_b19 --saved
  nice -n 19 "$PYTHON" -u -m scripts.verify_live_b18c --source "$source" --output-root logs/live_b19
done
for boundary in 2 3; do
  for stage in advance batch; do
    nice -n 19 "$PYTHON" -u -m scripts.verify_live_b19 --boundary "$boundary" --stage "$stage"
  done
done
touch logs/live_b19/verification.done
