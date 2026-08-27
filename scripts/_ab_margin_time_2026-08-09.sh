#!/usr/bin/env bash
# マージンタイム逓減 (--enable-margin-time-rate 相当) の認識 A/B。
# おじゃま判定の閾値が 70 点固定だったため、長い試合の後半で着弾を見逃していた
# (npz ベース実測: 148動画・15,347試合で **着弾の 6.27% を見逃し**)。
# 実際に認識を回して OJAMA_FALL の発火回数がどれだけ増えるかを確認する。
# 見逃しが多かった動画から選ぶ (c84=145件, c138=116件, c27=104件)。
set -u
ROOT=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
cd "$ROOT" || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=data/verify/margin_time_ab_2026-08-09
mkdir -p "$OUT"
for VID in c84 c138 c27; do
  SRC="$HOME/frames/video_${VID}.mp4"
  [ -f "$SRC" ] || { echo "skip $VID (no video)"; continue; }
  for MODE in off on; do
    EXTRA=""
    [ "$MODE" = "on" ] && EXTRA="--margin-time-rate"
    nice -n 19 ./venv/bin/python -u -m scripts._collect_lean_1t       --video "$SRC" --out-npz "$OUT/${VID}_${MODE}.npz"       --enable-chain-tracker --with-next --enable-effect-gate       --enable-burst-guard-v2 --enable-transition-merge-guard       --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard       --enable-match-transition-debounce --max-sec 600 --sample-interval 0       $EXTRA > "logs/margin_ab_${VID}_${MODE}.log" 2>&1 &
  done
done
wait
echo "MARGIN_AB_DONE $(date)"
ls -la "$OUT"
