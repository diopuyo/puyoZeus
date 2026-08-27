#!/bin/bash
# パネルレイアウト (--layout panel、2026-08-10 user指示) の確認用60秒サンプル生成。
# 構成は src/production_config.py の採用フラグ (デモ本番と同一) + counter-reach。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/video_c96.mp4
OUTDIR=data/verify/youtube_demo_2026-08-07/release
mkdir -p $OUTDIR
ADV=$(./venv/bin/python -c "from src.production_config import advantage_overlay_flags; print(advantage_overlay_flags())")
echo "[config] ADV: $ADV --counter-reach --layout panel"
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay --video $IN $ADV --counter-reach --layout panel --start-sec 1200 --max-sec 60 --warmup-sec 15 --out $OUTDIR/sample_panel_layout_60s.mp4"
{ echo "[cmd] $CMD"; time eval "$CMD"; } > logs/panel_layout_sample_2026-08-10.log 2>&1
echo "PANEL_SAMPLE_DONE $(date)"
ls -lh $OUTDIR/sample_panel_layout_60s.mp4
