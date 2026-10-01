#!/bin/bash
# コードはこのworktree、依存資産はセット1と同じexevを読み取る。
set -u
W=/mnt/d/puyo_analyzer/wt_evalset
E=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
O=$W/logs/eval_set/set2
R=$O/runtime
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p "$R" "$O/jobs"
for item in src scripts; do [ -e "$R/$item" ] || ln -sT "$W/$item" "$R/$item"; done
for item in data models logs; do [ -e "$R/$item" ] || ln -sT "$E/$item" "$R/$item"; done
cd "$R"
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
kind=$1
name=$2
if [ "$kind" = collect ]; then
  "$PY" -B -m scripts.collect_eval_set_20261001 --part "$name" --start-sec "$3" --end-sec "$4" --out "$O/collect" > "$O/jobs/${kind}_${name}.log" 2>&1
  status=$?
else
  variant=$3
  if [ "$kind" = regression ]; then
    record=$E/logs/pending_expiry/full/records/$name.jsonl.gz
  else
    record=$O/collect/records/$name.jsonl.gz
  fi
  extra=()
  [ "$variant" = e19 ] && extra=(--landing-counter-prob)
  "$PY" -B -m scripts.replay_exchange_event_20260926 "$record" --out "$O/$kind/$variant/$name" --production-exchange-event "${extra[@]}" > "$O/jobs/${kind}_${name}_${variant}.log" 2>&1
  status=$?
  name=${name}_${variant}
fi
echo "$status" > "$O/jobs/${kind}_${name}.done"
