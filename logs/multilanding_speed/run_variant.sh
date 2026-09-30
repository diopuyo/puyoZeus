#!/bin/bash
# 段2/段3: 計装付き再生を1プロセス・nice19で順次実行する。
# 使い方: run_variant.sh <ROOT(実行コードの置き場)> <OUT名> <mode: sources|live> [名前...]
#   sources: 保存記録5本 (q review fc mia zenchi) を再生 / live: リアルタイム実行が保存した inputs を再生
ROOT=$1; NAME=$2; MODE=$3; shift 3
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
RT=/mnt/c/Users/ryouj/.codex/worktrees/rt/puyo_analyzer/logs
OUT=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed/$NAME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $ROOT
mkdir -p $OUT
if [ "$MODE" = sources ]; then
  for s in "$@"; do
    nice -n 19 $PY -B -m scripts.profile_multilanding_speed --head-modules --source $s --out $OUT > $OUT/run_$s.log 2>&1
  done
else
  for s in "$@"; do
    case $s in
      b20_on) rec=$RT/live_b20/rt_on/candidate/inputs.jsonl.gz;;
      b18_run5) rec=$RT/live_b18/run5/candidate/inputs.jsonl.gz;;
      b18_stall) rec=$RT/live_b18/stall_stack/candidate/inputs.jsonl.gz;;
      b18_run20fix) rec=$RT/live_b18/run20fix/candidate/inputs.jsonl.gz;;
      b19_zenchi) rec=$RT/live_b19/video/zenchi/inputs.jsonl.gz;;
    esac
    nice -n 19 $PY -B -m scripts.profile_multilanding_speed --head-modules --variant ${VARIANT:-exact} --source live_$s --record $rec --out $OUT > $OUT/run_live_$s.log 2>&1
  done
fi
touch $OUT/DONE_$MODE
