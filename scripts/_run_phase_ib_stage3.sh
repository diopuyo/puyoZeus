#!/usr/bin/env bash
# Phase I.b Stage 3: ablation fine-tune + v29/v89 vanilla viz 並列起動
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p logs

# 1) ablation fine-tune (B-2 augment + S-7 topo-filter)
setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -u -m scripts.phase_i_fine_tune \
  --component cell_color \
  --all \
  --cell-save-to models/cnn_phase_b_finetuned_aug_topo.pt \
  --epochs 3 \
  --augment \
  --enable-topo-filter \
  > logs/phase_ib_finetune_aug_topo.log 2>&1 < /dev/null"
echo "launched ablation fine-tune"

# 2) v29 vanilla viz
setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -u -m scripts.visualize_recognition \
  --video data/evaluation_videos/v29_match2_156s.mp4 \
  --output data/evaluation_videos/v29_match2_phase_i_viz.mp4 \
  --cnn-model models/cnn_phase_b_finetuned.pt \
  > logs/phase_ib_viz_v29.log 2>&1 < /dev/null"
echo "launched v29 viz"

# 3) v89 vanilla viz
setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -u -m scripts.visualize_recognition \
  --video data/evaluation_videos/v89_match3_95s.mp4 \
  --output data/evaluation_videos/v89_match3_phase_i_viz.mp4 \
  --cnn-model models/cnn_phase_b_finetuned.pt \
  > logs/phase_ib_viz_v89.log 2>&1 < /dev/null"
echo "launched v89 viz"

sleep 3
echo ""
echo "=== alive procs ==="
pgrep -af 'phase_i_fine_tune|visualize_recognition' | grep -v grep | head -10
