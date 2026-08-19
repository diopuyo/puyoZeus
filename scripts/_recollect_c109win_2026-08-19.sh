#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
CF=$(./venv/bin/python -c "from src.production_config import collect_flags; print(collect_flags())")
PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_lean_1t \
  --video data/frames/video_c109.mp4 \
  --out-npz data/indicators_v2/boards_lean_lockfix_2026-08-19/c109_win1400.npz \
  $CF \
  --enable-lockdown-score-numeric-release \
  --enable-lockdown-score-moving-release \
  --enable-boundary-newmatch-evidence \
  --with-next --enable-phantom-board-guard \
  --start-sec 1400 --max-sec 560 --sample-interval 0
