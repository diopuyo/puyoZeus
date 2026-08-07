#!/bin/bash
# --overlay-transition-hold-frames 無し (対照実験用)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v2_2026-08-07.pt
nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition \
  --video "$IN" \
  --output data/verify/youtube_demo_2026-08-07/_smoke_nohold_control.mp4 \
  --overlay-show-states stable,tsumo_fall,gravity_settle \
  --cnn-model "$MODEL" \
  --max-sec 45 \
  > logs/smoke_nohold_control_2026-08-07.log 2>&1
echo "SMOKE_NOHOLD_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/_smoke_nohold_control.mp4
