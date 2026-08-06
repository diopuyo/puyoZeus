#!/bin/bash
# c19差分実験: new=59退行の犯人切り分け (2026-08-05、使い捨て)
# 3構成 (いずれも --max-sec 400 の部分走行、c19 t=332.5 の検証用):
#   A: v2_full相当 (§12なし・1.5bなし) = 退行前の健全基準
#   B: v3b - 連鎖延長なし (cooldownと1.5bのみ)
#   C: v3b - 1.5bなし (§12フルのみ)
# v3b本体 (全部あり) は退行確認済みのため再走行不要
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
LOG="logs/diff_experiment_2026-08-05.log"
mkdir -p data/verify/burst_guard_2026-08-05/diffexp logs
BASE="./venv/bin/python -u -m scripts.collect_boards_lean --video /home/ryouj/frames/video_c19.mp4 --max-sec 400 --enable-chain-tracker --with-next --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954"

echo "[diffexp] 開始 $(date)" >> "${LOG}"
nice -n 19 $BASE --out-npz data/verify/burst_guard_2026-08-05/diffexp/c19_A_v2full.npz >> "${LOG}" 2>&1 &
nice -n 19 $BASE --enable-hidden-row-burst-guard --out-npz data/verify/burst_guard_2026-08-05/diffexp/c19_B_no_ext.npz >> "${LOG}" 2>&1 &
nice -n 19 $BASE --enable-burst-close-extension --out-npz data/verify/burst_guard_2026-08-05/diffexp/c19_C_no_hrb.npz >> "${LOG}" 2>&1 &
wait
echo "[diffexp] ALL DONE $(date)" >> "${LOG}"
