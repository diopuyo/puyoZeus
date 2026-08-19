#!/bin/bash
# 試合境界の 0 リセット要求 (--enable-score-reset-requires-zero) の A/B 収集。
# 39番/38番を新フラグ込みで収集し、フラグ無しの boards_lean_wirecheck_2026-08-20
# と同一動画で対等比較する (memory feedback_paired_comparison_fixed_population)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/zeroreset_ab_2026-08-20.log 2>&1
echo "=== zeroreset A/B start $(date +%F_%T) ==="

CF=$(PYTHONPATH=. ./venv/bin/python -c "from src.production_config import collect_flags; print(collect_flags())")
OUT=data/indicators_v2/boards_lean_zeroreset_2026-08-20
mkdir -p "$OUT"

for T in 39 38; do
  echo "--- $T start $(date +%T) ---"
  PYTHONPATH=. ./venv/bin/python -m scripts._collect_lean_1t \
    --video "data/frames/video_${T}.mp4" \
    --out-npz "${OUT}/${T}.npz" \
    $CF \
    --enable-score-reset-requires-zero \
    --with-next --enable-phantom-board-guard \
    --max-sec 0 --sample-interval 0 \
    > "logs/zeroreset_${T}_2026-08-20.log" 2>&1
  echo "--- $T done rc=$? $(date +%T) ---"
done
echo "=== zeroreset A/B end $(date +%F_%T) ==="
