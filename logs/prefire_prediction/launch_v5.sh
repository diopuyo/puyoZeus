#!/bin/bash
# 指定worktree以外へ書かず、同時実行とメモリを制限する起動口。
set -euo pipefail
cd /mnt/d/puyo_analyzer/wt_prefire
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
if [[ "${1:-}" != run ]]; then exit 2; fi
shift
AVAILABLE=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
if (( AVAILABLE < 4194304 )); then echo 'available memory < 4GiB'; exit 3; fi
COUNT=$(pgrep -fc '[p]ython.*(scripts.prefire_v5|scripts.run_prefire_replay)' || true)
if (( COUNT >= 6 )); then echo 'six verification processes already running'; exit 4; fi
mkdir -p logs/prefire_prediction/v5_experiment
case "$1" in
  ledger) setsid -f bash -c 'exec nice -n 10 "$0" -B -m scripts.prefire_v5_ledger_samples > logs/prefire_prediction/v5_experiment/ledger.log 2>&1 < /dev/null' "$PY" ;;
  ledger_experiment) setsid -f bash -c 'exec nice -n 10 "$0" -B -m scripts.prefire_v5_experiment --ledger > logs/prefire_prediction/v5_experiment/ledger_experiment.log 2>&1 < /dev/null' "$PY" ;;
  short) setsid -f bash -c 'exec nice -n 10 "$0" -B -m scripts.prefire_v5_short_replay > logs/prefire_prediction/v5_experiment/short.log 2>&1 < /dev/null' "$PY" ;;
  smoke) setsid -f bash -c 'exec nice -n 10 "$0" -B -m scripts.prefire_v5_experiment --collect --limit 1 > logs/prefire_prediction/v5_experiment/smoke.log 2>&1 < /dev/null' "$PY" ;;
  experiment) setsid -f bash -c 'exec nice -n 10 "$0" -B -m scripts.prefire_v5_experiment > logs/prefire_prediction/v5_experiment/run.log 2>&1 < /dev/null' "$PY" ;;
  replay) setsid -f bash -c 'exec nice -n 10 "$0" -B -m scripts.run_prefire_replay_20260930 --variant v5 --source "$1" --latency 0.3 --tag _L03 > "logs/prefire_prediction/replay/v5_L03_$1.log" 2>&1 < /dev/null' "$PY" "$2" ;;
  *) exit 2 ;;
esac
