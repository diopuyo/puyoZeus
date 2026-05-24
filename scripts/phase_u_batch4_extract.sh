#!/bin/bash
set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
VIDEO="data/frames/video_01.mp4"
OUT_BASE="data/verify/phase_u_batch4"
mkdir -p "$OUT_BASE"

# 試合 (バッチ 1-3 未使用): m31, m33, m34, m35, m37, m39, m40, m42, m44, m46
declare -a MATCHES=(
    "31 1820 45"
    "33 1923 50"
    "34 1976 60"
    "35 2038 50"
    "37 2143 55"
    "39 2257 50"
    "40 2310 60"
    "42 2425 50"
    "44 2528 50"
    "46 2630 60"
)

for entry in "${MATCHES[@]}"; do
    read -r idx start dur <<< "$entry"
    t1=$(awk "BEGIN { printf \"%.1f\", $start + $dur * 0.30 }")
    t2=$(awk "BEGIN { printf \"%.1f\", $start + $dur * 0.50 }")
    t3=$(awk "BEGIN { printf \"%.1f\", $start + $dur * 0.70 }")
    t4=$(awk "BEGIN { printf \"%.1f\", $start + $dur * 0.85 }")
    bg_t=$(awk "BEGIN { printf \"%.1f\", $start - 2 }")
    out="$OUT_BASE/m${idx}"

    echo "=== match $idx ==="
    PYTHONPATH=. $PY -m scripts.phase_u_extract_samples \
        "$VIDEO" \
        --times "$t1,$t2,$t3,$t4" \
        --out-dir "$out" \
        --max-samples 50 \
        --side both \
        --bg-fp-time "$bg_t" \
        --skip-anim 2>&1 | tail -1
done

echo "=== batch 4 complete ==="
ls "$OUT_BASE/"
