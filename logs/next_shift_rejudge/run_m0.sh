#!/bin/bash
# M0 seed1..4 × orig/T (初回は CUDA 無効化 import で失敗したため再実行)
cd /mnt/d/puyo_analyzer/wt_nextfix
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
R=logs/next_shift_rejudge
for k in 1 2 3 4; do for v in orig T; do
  PYTHONPATH=. $PY -m scripts.next_shift_rejudge_models_20261001 --variant $v --seed $k --part m0 >> $R/m0_retry.log 2>&1 || echo "FAIL $v $k m0" >> $R/m0_retry.log
done; done
echo done > $R/m0_retry.done
