#!/bin/bash
# Phase U batch 2: 新ロジック適用 (UI Mask 強化 + skip-anim + 赤H拡張)
set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
VIDEO="data/frames/video_01.mp4"
OUT_BASE="data/verify/phase_u_batch2"
mkdir -p "$OUT_BASE"

# 試合定義 (idx, start_sec, duration_sec) - 第 1 弾と異なる試合を選定
declare -a MATCHES=(
    "4 383 30"
    "5 416 33"
    "7 502 39"
    "15 972 90"
    "16 1065 50"
    "17 1118 45"
    "19 1190 49"
    "20 1242 53"
    "23 1378 57"
    "25 1478 74"
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
        --skip-anim 2>&1 | tail -2
done

echo ""
echo "=== batch 2 complete: $OUT_BASE/ ==="
ls "$OUT_BASE/"
