#!/bin/bash
# パネルレイアウト (字幕余白入り、--layout panel) の全編本番生成 (2026-08-10 user指示)。
# 構成は src/production_config.py の採用フラグ (デモ本番と同一) + counter-reach。
# 入力は生の保持動画 (5時間30分、60fps) をそのまま渡す (切り出しなし=「全編」)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/video_c96.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07/release
mkdir -p $OUTDIR
ADV=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
echo "[config] ADV: $ADV --counter-reach --layout panel"
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV --counter-reach --layout panel --out $OUTDIR/final_5_panel_layout.mp4"
echo "[cmd] $CMD"
eval "$CMD"
echo "PANEL_FULL_DONE $(date)"
ls -lh $OUTDIR/final_5_panel_layout.mp4
