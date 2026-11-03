#!/bin/bash
# 先に短区間を逐次実行し、全体再生はこのスクリプトから起動しない。
set -eu
ROOT=/mnt/d/puyo_analyzer/wt_evalset
cd "$ROOT"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mode="${1:-v2}"
case "$mode" in v2|v2b) ;; *) exit 2;; esac
if [ "$mode" = v2b ]; then
  export FALSEDEATH_POLL_SECONDS=1
  export FALSEDEATH_CONTROLLER_MODULES='scripts.plan_c_t_memory_watch scripts.gfe_phase_stage2_driver_20261002'
fi
OUT="$ROOT/logs/eval_set/falsedeath_fix/$mode"
mkdir -p "$OUT"
worker_log="${FALSEDEATH_WORKER_LOG:-ladder_worker.log}"
test ! -e "$OUT/$worker_log"
exec > "$OUT/$worker_log" 2>&1
source "$ROOT/logs/resource_falsedeath.sh"
for source in c5 mia8KCjr52g review; do
  echo "START $source $(date -Iseconds)"
  run_limited "$OUT/short_$source.log" nice -n 10 "$PY" -B -m scripts.falsedeath_v2_ladder_20261002 "$source" --mode "$mode"
  echo "DONE $source $(date -Iseconds)"
done
run_limited "$OUT/ladder_score.log" nice -n 10 "$PY" -B -m scripts.falsedeath_v2_ladder_20261002 score --mode "$mode"
echo "LADDER_DONE $(date -Iseconds)"
