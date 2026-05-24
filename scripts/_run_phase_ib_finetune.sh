#!/usr/bin/env bash
# Phase I.b: cell color CNN fine-tune (10 動画 cell.jsonl 累計 1.6M+ samples)
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -u -m scripts.phase_i_fine_tune \
  --component cell_color \
  --all \
  --cell-save-to models/cnn_phase_b_finetuned.pt \
  --epochs 3 \
  > logs/phase_ib_finetune.log 2>&1 < /dev/null"

sleep 2
echo "launched. log: logs/phase_ib_finetune.log"
pgrep -af 'phase_i_fine_tune' | grep -v grep | head -5
