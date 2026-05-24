#!/bin/bash
# cycle_19 (真 baseline) vs cycle_26 (= A1+A2+A4) を 5 動画 side-by-side 比較
# 完走後すぐ目視レビュー可能にする。
# 出力: data/test_unknown/compare_19_vs_26_<tag>.mp4 × 5
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
OUT_DIR=data/test_unknown

for tag in v50 v70 v89m3 v91 v97; do
    LEFT="$OUT_DIR/${tag}_viz_multicycle_19.mp4"
    RIGHT="$OUT_DIR/${tag}_viz_multicycle_26.mp4"
    OUTPUT="$OUT_DIR/compare_19_vs_26_${tag}.mp4"
    if [ ! -f "$LEFT" ]; then
        echo "[skip] $tag: cycle_19 viz not found ($LEFT)"
        continue
    fi
    if [ ! -f "$RIGHT" ]; then
        echo "[skip] $tag: cycle_26 viz not found ($RIGHT)"
        continue
    fi
    echo "[compare] $tag : $LEFT (left=cycle_19) vs $RIGHT (right=cycle_26)"
    PYTHONPATH=. ./venv/bin/python -m scripts.cycle_compare_video \
        --left "$LEFT" --right "$RIGHT" \
        --output "$OUTPUT" 2>&1 | tail -3
done

echo
echo "[done] 比較動画一覧:"
ls -lh $OUT_DIR/compare_19_vs_26_*.mp4 2>/dev/null
