#!/bin/bash
# D-v2h2b専用の資源制限起動口。既存成果物は上書きしない。
set -eu
ROOT=/mnt/d/puyo_analyzer/wt_evalset
cd "$ROOT"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FALSEDEATH_POLL_SECONDS=1 NUMEXPR_NUM_THREADS=1
# いずれもsubprocess待機だけの制御器。子の計算モジュールは引き続き全て数える。
export FALSEDEATH_CONTROLLER_MODULES='scripts.plan_c_t_memory_watch scripts.gfe_phase_stage2_driver_20261002 scripts.plan_c_t_controls scripts._diag_gfe_stage2_capture_all scripts._diag_gfe_stage2_finish scripts.gfe_phase_stage2b_driver_20261002 scripts.gfe_phase_stage2b_monitor_20261002'
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT="$ROOT/logs/eval_set/falsedeath_fix/v2h2b"
mkdir -p "$OUT"
source "$ROOT/logs/resource_falsedeath.sh"
name="$1"
shift
case "$name" in
  full_*)
    # 同じ記録を複数の制御器が要求しても、再生は一度だけにする。
    exec 8> "$OUT/$name.lock"
    flock 8
    if [ -f "$ROOT/logs/eval_set/falsedeath_fix/replay/v2h2b/${name#full_}/status.json" ]; then
      exit 0
    fi
    ;;
esac
test ! -e "$OUT/$name.log"
run_limited "$OUT/$name.log" nice -n 10 "$PY" -B "$@"
