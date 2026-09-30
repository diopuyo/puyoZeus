#!/bin/bash
# ゲート再生 (保存記録 5 本 → 採点)。使い方: run_gate.sh <variant: exact|bounded|bounded2> ; nice19・1 プロセス・順次
V=$1
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=logs/multilanding_speed/gate_$V
mkdir -p $OUT
for s in q_7gc4TgFig review fcXG83vInDY mia8KCjr52g zenchi; do
  nice -n 19 $PY -B -m scripts.run_multilanding_speed_gate --variant $V --step replay --source $s > $OUT/replay_$s.log 2>&1
done
nice -n 19 $PY -B -m scripts.run_multilanding_speed_gate --variant $V --step report > $OUT/report.log 2>&1
touch $OUT/GATE_DONE
