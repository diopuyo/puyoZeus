#!/bin/bash
# 公開用 最終2本 (2026-08-09 user確定・下部オーバーレイなし)。
#   final_1_full_overlay.mp4 : 連鎖会計の窓 (おじゃま予告パネル+優勢バー) あり
#   final_2_stable_tsumo.mp4  : ツモ落下中とSTABLEの間だけオーバーレイを出す
# 下部のおじゃま予告パネル+優勢バーは user 指示で非表示 (--hide-ojama-forecast)。
# 構成は src/production_config.py から取得 (手書きしない)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07/release
mkdir -p $OUTDIR
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt
REC=$(./venv/bin/python -c "from src.production_config import recognition_flags; print(recognition_flags())")
VIZ=$(./venv/bin/python -c "from src.production_config import visualization_flags; print(visualization_flags())")
echo "[config] REC: $REC"
echo "[config] VIZ: $VIZ"

BASE="--video $IN --cnn-model $MODEL $REC $VIZ --enable-asymmetric-recovery-min-frames --recovery-add-min-frames 3 --overlay-cell-stability-frames 14 --hide-ojama-forecast --enable-ojama-entry-gravity-settle-guard --enable-gravity-settle-reset-on-exit --overlay-show-chain-count"

C1="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition $BASE --output $OUTDIR/final_1_full_overlay.mp4"
C2="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition $BASE --overlay-show-states stable,tsumo_fall --overlay-state-debounce-frames 10 --output $OUTDIR/final_2_stable_tsumo.mp4"

{ echo "[cmd] $C1"; eval "$C1"; } > logs/final_1_2026-08-09.log 2>&1 &
{ echo "[cmd] $C2"; eval "$C2"; } > logs/final_2_2026-08-09.log 2>&1 &
wait
echo "FINAL2_DONE $(date)"
ls -lh $OUTDIR/final_*.mp4
