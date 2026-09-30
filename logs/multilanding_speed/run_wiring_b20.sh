#!/bin/bash
# 隠し段候補上限が実際に効く記録 (rt b20、43通知で発動) で、--production-exchange-event だけの再生が
# B2 明示指定の再生 (live_bounded2) と display / events で完全一致するかを確かめる。nice19・1 プロセス。
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
O=logs/multilanding_speed/wiring_check/b20
mkdir -p $O
nice -n 19 $PY -B -m scripts.replay_exchange_event_20260926 /mnt/c/Users/ryouj/.codex/worktrees/rt/puyo_analyzer/logs/live_b20/rt_on/candidate/inputs.jsonl.gz \
  --out $O --production-exchange-event --compare logs/multilanding_speed/live_bounded2/on/live_b20_on > $O.log 2>&1
echo "exit=$?" >> $O.log
