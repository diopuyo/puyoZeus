#!/bin/bash
# 幽霊セル対策 (--overlay-transition-hold-frames) スモーク確認用
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v2_2026-08-07.pt
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/_smoke_hold.mp4 \
  --overlay-show-states stable,tsumo_fall,gravity_settle \
  --overlay-transition-hold-frames 30 \
  --cnn-model "$MODEL" \
  --max-sec 45 \
  > logs/smoke_hold_2026-08-07.log 2>&1
echo "SMOKE_HOLD_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/_smoke_hold.mp4
