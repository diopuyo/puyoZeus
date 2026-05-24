#!/bin/bash
# Phase U: 1000 件ラベル付け第 1 弾 (v01 from 10 matches = 500 cells)
set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
VIDEO="data/frames/video_01.mp4"
OUT_BASE="data/verify/phase_u_batch1"
mkdir -p "$OUT_BASE"

# 試合定義 (idx, start_sec, duration_sec)
declare -a MATCHES=(
    "1 186 70"
    "2 259 60"
    "3 321 60"
    "6 452 47"
    "8 544 79"
    "9 625 61"
    "10 688 61"
    "11 751 99"
    "13 882 48"
    "14 932 38"
)

for entry in "${MATCHES[@]}"; do
    read -r idx start dur <<< "$entry"
    # 試合長の 30%, 50%, 70%, 85% を使う
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
        --bg-fp-time "$bg_t" 2>&1 | tail -2
done

echo ""
echo "=== batch 1 complete: $OUT_BASE/ ==="
ls "$OUT_BASE/"
