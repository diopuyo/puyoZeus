#!/bin/bash
# 診断・合否外。指定worktreeの新規出力だけに書き込む。
set -eu
ROOT=/mnt/d/puyo_analyzer/wt_evalset
cd "$ROOT"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FALSEDEATH_POLL_SECONDS=1
export FALSEDEATH_CONTROLLER_MODULES='scripts.plan_c_t_memory_watch scripts.gfe_phase_stage2_driver_20261002'
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT="$ROOT/logs/eval_set/falsedeath_fix/v2b_ablate"
mkdir -p "$OUT"
worker_log="${FALSEDEATH_WORKER_LOG:-worker.log}"
test ! -e "$OUT/$worker_log"
exec > "$OUT/$worker_log" 2>&1
source "$ROOT/logs/resource_falsedeath.sh"
for mode in A B control; do
  echo "START $mode $(date -Iseconds)"
  run_limited "$OUT/$mode.log" nice -n 10 "$PY" -B -m scripts._diag_falsedeath_v2b_ablate "$mode"
  echo "DONE $mode $(date -Iseconds)"
done
