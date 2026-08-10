#!/bin/bash
# 映像C / D のみを生成する (A/B が別途走行中のときに使う)。
# 構成は src/production_config.py から取得する (手書きしない)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt
REC_FLAGS=$(./venv/bin/python -c "from src.production_config import recognition_flags; print(recognition_flags())")
VIZ_FLAGS=$(./venv/bin/python -c "from src.production_config import visualization_flags; print(visualization_flags())")
echo "[config] REC: $REC_FLAGS"
echo "[config] VIZ: $VIZ_FLAGS"

BASE="--video $IN --cnn-model $MODEL $REC_FLAGS $VIZ_FLAGS --enable-asymmetric-recovery-min-frames --recovery-add-min-frames 3 --overlay-cell-stability-frames 14 --hide-ojama-forecast --enable-ojama-entry-gravity-settle-guard --enable-gravity-settle-reset-on-exit --overlay-show-chain-count"

CMD_C="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition $BASE --output $OUTDIR/demo_final_C_recognition.mp4"
CMD_D="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition $BASE --hide-cell-overlay --output $OUTDIR/demo_final_D_no_cell_overlay.mp4"

{ echo "[cmd] $CMD_C"; eval "$CMD_C"; } > logs/demo_final_C_2026-08-08.log 2>&1 &
{ echo "[cmd] $CMD_D"; eval "$CMD_D"; } > logs/demo_final_D_2026-08-08.log 2>&1 &
wait
echo "DEMO_CD_DONE $(date)"
ls -lh $OUTDIR/demo_final_C_recognition.mp4 $OUTDIR/demo_final_D_no_cell_overlay.mp4
