#!/bin/bash
# 段2: 保存記録の再生を計装付きで1プロセス・nice19で順次実行 (出力は logs/multilanding_speed/base)
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=${OUT:-logs/multilanding_speed/base}
mkdir -p $OUT
for s in "$@"; do
  nice -n 19 $PY -B -m scripts.profile_multilanding_speed --source $s --out $OUT > $OUT/run_$s.log 2>&1
done
touch $OUT/PROFILE_DONE
