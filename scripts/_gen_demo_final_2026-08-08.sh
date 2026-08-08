#!/bin/bash
# YouTube デモ 最終版 — **本番構成 (src/production_config.py) から
# フラグを自動生成する**。 手でフラグを並べないことが本スクリプトの要点で、
# 2026-08-08 の退行 (--early-fire-reaction 付け忘れ) の再発防止策。
#
# 生成物:
#   demo_final_A_advantage.mp4        映像A: 有利不利のみ
#   demo_final_B_with_recognition.mp4 映像B: A + 認識オーバーレイ
#   demo_final_C_recognition.mp4      映像C: 認識のみ (色記号 + 状態 + 連鎖数)
#   demo_final_D_no_cell_overlay.mp4  映像D: 色記号なし (状態 + 連鎖数のみ)
# いずれも同一試合 (dio vs TS m01)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt

# --- 本番構成を単一の情報源から取得 (手書きしない) ---
ADV_FLAGS=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
REC_FLAGS=$(./venv/bin/python -c "from src.production_config import recognition_flags; print(recognition_flags())")
VIZ_FLAGS=$(./venv/bin/python -c "from src.production_config import visualization_flags; print(visualization_flags())")
echo "[config] ADV: $ADV_FLAGS"
echo "[config] REC: $REC_FLAGS"
echo "[config] VIZ: $VIZ_FLAGS"
# 生成物と一緒に構成を残す (どの構成で作ったか後から辿れるように)
./venv/bin/python -m src.production_config > "$OUTDIR/demo_final_config_2026-08-08.txt"

# 映像A / B: 有利不利オーバーレイ (認識は pipeline 既定 + 採用済み)
CMD_A="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV_FLAGS --out $OUTDIR/demo_final_A_advantage.mp4"
CMD_B="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV_FLAGS --show-recognition --out $OUTDIR/demo_final_B_with_recognition.mp4"
# 映像C: 認識オーバーレイ (デモ専用 CNN + 表示系の採用済みフラグ)
CMD_C="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition --video $IN --output $OUTDIR/demo_final_C_recognition.mp4 --cnn-model $MODEL $REC_FLAGS $VIZ_FLAGS --enable-asymmetric-recovery-min-frames --recovery-add-min-frames 3 --overlay-cell-stability-frames 14 --hide-ojama-forecast --enable-ojama-entry-gravity-settle-guard --enable-gravity-settle-reset-on-exit --overlay-show-chain-count"
# 映像D: 映像C から盤面セルの色記号だけ落とした版 (user要望)。
# 状態ラベル・連鎖数・枠は残るので「認識が追従していること」は見える。
CMD_D="${CMD_C/demo_final_C_recognition/demo_final_D_no_cell_overlay} --hide-cell-overlay"

{ echo "[cmd] $CMD_A"; eval "$CMD_A"; } > logs/demo_final_A_2026-08-08.log 2>&1 &
{ echo "[cmd] $CMD_B"; eval "$CMD_B"; } > logs/demo_final_B_2026-08-08.log 2>&1 &
{ echo "[cmd] $CMD_C"; eval "$CMD_C"; } > logs/demo_final_C_2026-08-08.log 2>&1 &
{ echo "[cmd] $CMD_D"; eval "$CMD_D"; } > logs/demo_final_D_2026-08-08.log 2>&1 &
wait
echo "DEMO_FINAL_DONE $(date)"
ls -lh $OUTDIR/demo_final_*.mp4
