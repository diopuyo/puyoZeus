#!/bin/bash
set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
VIDEO="data/frames/video_03.mp4"
OUT_BASE="data/verify/phase_u_v03"
mkdir -p "$OUT_BASE"

# video_03 (46 matches, 720p->1080p resize)
# Pick 6 matches spread across the tournament for diversity:
# early (m1, m10), middle (m18, m27), late (m36, m45)
declare -a MATCHES=(
    "1 185 73"
    "10 831 85"
    "18 1480 50"
    "27 2021 51"
    "36 2514 35"
    "45 2911 64"
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

echo "=== video_03 batch complete ==="
ls "$OUT_BASE/"
