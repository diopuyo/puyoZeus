#!/usr/bin/env bash
# 使い方: _run_cli_identity.sh <prod|explicit> <source...>  再生CLIを本番指定/明示指定で実行 (nice 0)
set -eu
cd /mnt/d/puyo_analyzer/wt_switch
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
REC=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/full/records
mode=$1; shift
if [ "$mode" = prod ]; then FLAGS="--production-exchange-event"; else
  FLAGS=$($PY -c "from src import production_config as c; print(c.exchange_event_flags())"); fi
for s in "$@"; do
  mkdir -p logs/switch_smoothing/cli_$mode/$s
  $PY -B -m scripts.replay_exchange_event_20260926 $REC/$s.jsonl.gz --out logs/switch_smoothing/cli_$mode/$s $FLAGS \
    > logs/switch_smoothing/cli_$mode/$s.log 2>&1
done
touch logs/switch_smoothing/cli_$mode/DONE_$1
