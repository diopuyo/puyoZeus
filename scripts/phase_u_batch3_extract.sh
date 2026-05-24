#!/bin/bash
# Phase U batch 3: 新規 10 試合 (合計 1500 件目標)
set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
VIDEO="data/frames/video_01.mp4"
OUT_BASE="data/verify/phase_u_batch3"
mkdir -p "$OUT_BASE"

# 試合定義 (バッチ 1, 2 未使用): m18, m21, m22, m24, m26, m27, m28, m29, m30, m32
declare -a MATCHES=(
    "18 1166 21"
    "21 1297 46"
    "22 1345 30"
    "24 1438 38"
    "26 1554 60"
    "27 1616 50"
    "28 1668 60"
    "29 1730 50"
    "30 1782 60"
    "32 1865 55"
)

for entry in "${MATCHES[@]}"; do
    read -r idx start dur <<< "$entry"
    t1=$(awk "BEGIN { printf \"%.1f\", $start + $dur * 0.30 }")
    t2=$(awk "BEGIN { printf \"%.1f\", $start + $dur * 0.50 }")
    t3=$(awk "BEGIN { printf \"%.1f\", $start + $dur * 0.70 }")
    t4=$(awk "BEGIN { printf \"%.1f\", $start + $dur * 0.85 }")
    bg_t=$(awk "BEGIN { printf \"%.1f\", $start - 2 }")
    out="$OUT_BASE/m${idx}"

    echo ""
    echo "=== match $idx (start=$start dur=$dur) -> $out ==="
    PYTHONPATH=. $PY -m scripts.phase_u_extract_samples \
        "$VIDEO" \
        --times "$t1,$t2,$t3,$t4" \
        --out-dir "$out" \
        --max-samples 50 \
        --side both \
        --bg-fp-time "$bg_t" \
        --skip-anim 2>&1 | tail -1
done

echo ""
echo "=== batch 3 complete: $OUT_BASE/ ==="
ls "$OUT_BASE/"
