#!/bin/bash
# 事前登録の梯子だけで全体再生への進行を決める。
set -eu
ROOT=/mnt/d/puyo_analyzer/wt_evalset
cd "$ROOT"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FALSEDEATH_POLL_SECONDS=1
export FALSEDEATH_CONTROLLER_MODULES='scripts.plan_c_t_memory_watch scripts.gfe_phase_stage2_driver_20261002'
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT="$ROOT/logs/eval_set/falsedeath_fix/v2b"
worker_log="${FALSEDEATH_FULL_LOG:-full_worker.log}"
test ! -e "$OUT/$worker_log"
exec > "$OUT/$worker_log" 2>&1
echo "WAIT_LADDER $(date -Iseconds)"
while [ ! -f "$OUT/LADDER.json" ]; do sleep 5; done
if ! nice -n 10 "$PY" -B -c 'import json,sys; sys.exit(not json.load(open(sys.argv[1]))["passed"])' "$OUT/LADDER.json"; then
  echo "STOP_LADDER_FAILED $(date -Iseconds)"
  exit 0
fi
echo "FULL_START $(date -Iseconds)"
bash logs/replay_falsedeath.sh v2b c1 c2 c3 c4 c5 c6 zenchi s0 s1 s2 s3 s4 s5 s6 review q_7gc4TgFig fcXG83vInDY mia8KCjr52g
source logs/resource_falsedeath.sh
run_limited "$OUT/full_score.log" nice -n 10 "$PY" -B -m scripts.falsedeath_score_20261001 --mode v2b
echo "FULL_SCORED $(date -Iseconds)"
