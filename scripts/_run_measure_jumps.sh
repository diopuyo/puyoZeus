#!/usr/bin/env bash
# 保存済み本番記録 (e36b_on) の切替時の飛びを測る (読取専用・1プロセス)
set -eu
cd /mnt/d/puyo_analyzer/wt_switch
S=<home>/.codex/worktrees/exev/puyo_analyzer/logs/pending_expiry/e36b_on/on
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
mkdir -p logs/switch_smoothing
PYTHONPATH=. nice -n 10 $PY -B -m scripts.measure_switch_jumps --out logs/switch_smoothing/baseline_jumps.json \
  --run zenchi=$S/zenchi --run review=$S/review \
  --run q=$S/renders/q_7gc4TgFig/on --run fc=$S/renders/fcXG83vInDY/on --run mia=$S/renders/mia8KCjr52g/on
