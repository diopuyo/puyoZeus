#!/bin/bash
# 配線検査: --production-exchange-event だけで再生し、B2 を明示指定した既存の門 run (gate_bounded2) と
# display.npz / events.jsonl が完全一致するか (--compare) を保存記録 5 本で確かめる。nice19・1 プロセス。
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
G=logs/multilanding_speed/gate_bounded2/on
O=logs/multilanding_speed/wiring_check
for s in q_7gc4TgFig review fcXG83vInDY mia8KCjr52g zenchi; do
  case $s in review|zenchi) ref=$G/$s;; *) ref=$G/renders/$s/on;; esac
  nice -n 19 $PY -B -m scripts.replay_exchange_event_20260926 logs/pending_expiry/full/records/$s.jsonl.gz \
    --out $O/$s --production-exchange-event --compare $ref > $O/$s.log 2>&1
  echo "$s exit=$?" >> $O/SUMMARY.txt
done
touch $O/DONE
