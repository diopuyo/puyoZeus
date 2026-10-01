#!/bin/bash
cd /mnt/d/puyo_analyzer/wt_nextfix
export OMP_NUM_THREADS=2
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
L=logs/next_shift_train
PYTHONPATH=. $PY -m scripts.next_shift_gfe_20261001 --m0-root $L/m0_T --output $L/gfe_T --models models/exchange_event_v5_common --stage final > $L/gfe_final.log 2>&1
echo "gfe $?" > $L/final_models.done
PYTHONPATH=. $PY -m scripts.train_exchange_event_m0 --inputs $L/m0_T/inputs --output models/exchange_event_v5_common/M0 > $L/m0_final.log 2>&1
echo "m0 $?" >> $L/final_models.done
