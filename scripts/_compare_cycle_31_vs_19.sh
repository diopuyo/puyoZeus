#!/bin/bash
# cycle_31 (新 = 左) vs cycle_19 (旧 = 右、真 baseline) を 5 動画 side-by-side
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
OUT_DIR=data/test_unknown

for tag in v50 v70 v89m3 v91 v97; do
    LEFT="$OUT_DIR/${tag}_viz_multicycle_31.mp4"
    RIGHT="$OUT_DIR/${tag}_viz_multicycle_19.mp4"
    OUTPUT="$OUT_DIR/compare_31_vs_19_${tag}.mp4"
    if [ ! -f "$LEFT" ] || [ ! -f "$RIGHT" ]; then
        echo "[skip] $tag: missing file"
        continue
    fi
    echo "[compare] $tag : cycle_31 (left=new) vs cycle_19 (right=old)"
    PYTHONPATH=. ./venv/bin/python -m scripts.cycle_compare_video \
        --left "$LEFT" --right "$RIGHT" \
        --output "$OUTPUT" 2>&1 | tail -3
done

echo
echo "[done] 比較動画一覧:"
ls -lh $OUT_DIR/compare_31_vs_19_*.mp4 2>/dev/null
