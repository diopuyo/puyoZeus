#!/bin/bash
set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
VIDEO="data/frames/video_02.mp4"
OUT_BASE="data/verify/phase_u_v02"
mkdir -p "$OUT_BASE"

# video_02 (50 matches, 720p->1080p resize)
# Pick 6 matches spread across the tournament for diversity:
# early (m1, m11), middle (m20, m30), late (m40, m49)
declare -a MATCHES=(
    "1 205 56"
    "11 739 34"
    "20 1180 38"
    "30 1778 33"
    "40 2320 47"
    "49 2828 119"
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

echo "=== video_02 batch complete ==="
ls "$OUT_BASE/"
