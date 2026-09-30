#!/bin/bash
# 遅い通知だけを cProfile で保存する (現行ツリー・HEAD 証明器、nice19・1 プロセス)
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=logs/multilanding_speed/cprof_b20
mkdir -p $OUT
nice -n 19 $PY -B -m scripts.profile_multilanding_speed --head-modules --source live_b20_on --record /mnt/c/Users/ryouj/.codex/worktrees/rt/puyo_analyzer/logs/live_b20/rt_on/candidate/inputs.jsonl.gz --out $OUT --cprofile-at 6492.8167,6245.5167,5984.6 > $OUT/run.log 2>&1
touch $OUT/DONE
