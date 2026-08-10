#!/bin/bash
# 陰性対照 v2 (2026-08-08): 試合を含む dio_vs_ts_warmup_clip.mp4 (370秒、
# 試合1〜7を含む) を新標準構成で regen → 完了後にゲートを自動実行する。
# v1 (olRyxDGacbg 先頭10分、試合0件) の誤PASS事故の修正版。
# 1プロセス・nice -19、既存 14並列 regen とは競合しない優先度で走らせる。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

VIDEO="data/verify/youtube_demo_2026-08-07/dio_vs_ts_warmup_clip.mp4"
OUT_DIR="data/verify/phase_l_quality_gate_2026-08-07/negative_control"
NPZ="${OUT_DIR}/dio_vs_ts_warmup_clip.npz"
LOG="logs/phase_l_quality_gate_negctrl_v2_2026-08-08.log"
mkdir -p "${OUT_DIR}"

echo "[negctrl_v2] regen 開始 $(date)" >> "${LOG}"
nice -n 19 ./venv/bin/python -u -m scripts._collect_lean_1t \
  --video "${VIDEO}" --out-npz "${NPZ}" \
  --enable-chain-tracker --with-next \
  --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard \
  --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard \
  --enable-match-transition-debounce \
  --max-sec 0 --sample-interval 0 >> "${LOG}" 2>&1
echo "[negctrl_v2] regen 完了 $(date)" >> "${LOG}"

if [ -f "${NPZ}" ]; then
  ./venv/bin/python scripts/phase_l_video_quality_gate.py \
    --video dio_vs_ts_warmup_clip --target-npz "${NPZ}" \
    --out-dir "${OUT_DIR}" >> "${LOG}" 2>&1
  echo "[negctrl_v2] gate 完了 $(date)" >> "${LOG}"
else
  echo "[negctrl_v2][ERROR] npz missing after regen" >> "${LOG}"
fi
