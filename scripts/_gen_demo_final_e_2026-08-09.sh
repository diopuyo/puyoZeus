#!/bin/bash
# 映像E: 認識オーバーレイを **確定盤面 (stable) とツモ落下のときだけ** 表示する版。
# 以前の FINAL*b 系に相当。 デモを A/B/C/D に再編した際に落としてしまったため復活
# (2026-08-09 user指摘)。
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
# 確定盤面 + ツモ落下のみ表示 (瞬間フリッカーでは消さないようデバウンス 10 フレーム)
STATES="--overlay-show-states stable,tsumo_fall --overlay-state-debounce-frames 10"

CMD_E="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition $BASE $STATES --output $OUTDIR/demo_final_E_stable_tsumo.mp4"

{ echo "[cmd] $CMD_E"; eval "$CMD_E"; } > logs/demo_final_E_2026-08-09.log 2>&1
echo "DEMO_E_DONE $(date)"
ls -lh $OUTDIR/demo_final_E_stable_tsumo.mp4
