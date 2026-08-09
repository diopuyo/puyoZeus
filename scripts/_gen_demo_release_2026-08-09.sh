#!/bin/bash
# YouTube 公開用デモ 最終生成 (2026-08-09)。
# 構成は src/production_config.py から取得する (手書きしない = 退行防止)。
# 採用済み: 早期発火 / 片側独立更新 / 得点タイブレーク除去 / 圧力除去 / Platt較正
#
# 生成物 (すべて同一試合 dio vs TS m01):
#   release_A_advantage.mp4        有利不利のみ
#   release_B_with_recognition.mp4 有利不利 + 認識オーバーレイ
#   release_C_recognition.mp4      認識のみ (色記号+状態+連鎖数)
#   release_D_no_cell_overlay.mp4  色記号なし (状態+連鎖数のみ)
#   release_E_stable_tsumo.mp4     確定盤面+ツモ落下のみ表示
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07/release
mkdir -p $OUTDIR
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt
ADV=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
REC=$(./venv/bin/python -c "from src.production_config import recognition_flags; print(recognition_flags())")
VIZ=$(./venv/bin/python -c "from src.production_config import visualization_flags; print(visualization_flags())")
echo "[config] ADV: $ADV"
echo "[config] REC: $REC"
echo "[config] VIZ: $VIZ"
./venv/bin/python -m src.production_config > $OUTDIR/config.txt

VBASE="--video $IN --cnn-model $MODEL $REC $VIZ --enable-asymmetric-recovery-min-frames --recovery-add-min-frames 3 --overlay-cell-stability-frames 14 --hide-ojama-forecast --enable-ojama-entry-gravity-settle-guard --enable-gravity-settle-reset-on-exit --overlay-show-chain-count"

A="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV --out $OUTDIR/release_A_advantage.mp4"
B="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV --show-recognition --out $OUTDIR/release_B_with_recognition.mp4"
C="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition $VBASE --output $OUTDIR/release_C_recognition.mp4"
D="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition $VBASE --hide-cell-overlay --output $OUTDIR/release_D_no_cell_overlay.mp4"
E="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition $VBASE --overlay-show-states stable,tsumo_fall --overlay-state-debounce-frames 10 --output $OUTDIR/release_E_stable_tsumo.mp4"

{ echo "[cmd] $A"; eval "$A"; } > logs/release_A_2026-08-09.log 2>&1 &
{ echo "[cmd] $B"; eval "$B"; } > logs/release_B_2026-08-09.log 2>&1 &
{ echo "[cmd] $C"; eval "$C"; } > logs/release_C_2026-08-09.log 2>&1 &
{ echo "[cmd] $D"; eval "$D"; } > logs/release_D_2026-08-09.log 2>&1 &
{ echo "[cmd] $E"; eval "$E"; } > logs/release_E_2026-08-09.log 2>&1 &
wait
echo "RELEASE_DONE $(date)"
ls -lh $OUTDIR/*.mp4
