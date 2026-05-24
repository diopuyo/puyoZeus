#!/bin/bash
# W4: video_04 〜 video_19 (pl5+pl6) で count_match_via_ocr + detect_match_winners
# を 4 並列で実行。完了後は data/verify/match_boundaries_v5/ と
# data/verify/match_winners_v0X.tsv が揃う。
#
# 使い方:
#   bash scripts/phase_w_batch_match_detect.sh

set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
PARALLELISM=4
LOG_DIR="/tmp/phase_w_batch"
mkdir -p "$LOG_DIR"

# 動画 04 〜 19 (16 動画)、video_10 はスキップ (既に done)
VIDEOS=(04 05 06 07 08 09 11 12 13 14 15 16 17 18 19)

run_count_match() {
    local vid=$1
    local out="data/verify/match_boundaries_v5/video_${vid}/matches.tsv"
    if [ -s "$out" ]; then
        echo "  [count] v${vid}: SKIP (already done)"
        return 0
    fi
    echo "  [count] v${vid}: START"
    PYTHONPATH=. $PY -m scripts.count_match_via_ocr \
        --video "data/frames/video_${vid}.mp4" \
        --out "$out" \
        --interval 1.0 --confirm 2 \
        > "$LOG_DIR/cm_${vid}.log" 2>&1
    if [ $? -eq 0 ]; then
        local n=$(($(wc -l < "$out") - 1))
        echo "  [count] v${vid}: DONE ($n matches)"
    else
        echo "  [count] v${vid}: FAILED"
    fi
}

run_detect_winners() {
    local vid=$1
    local matches="data/verify/match_boundaries_v5/video_${vid}/matches.tsv"
    local out="data/verify/match_winners_v${vid}.tsv"
    if [ ! -s "$matches" ]; then
        echo "  [winner] v${vid}: SKIP (no matches.tsv)"
        return 0
    fi
    if [ -s "$out" ]; then
        echo "  [winner] v${vid}: SKIP (already done)"
        return 0
    fi
    echo "  [winner] v${vid}: START"
    PYTHONPATH=. $PY scripts/detect_match_winners.py \
        --video "data/frames/video_${vid}.mp4" \
        --matches-tsv "$matches" \
        --out "$out" \
        > "$LOG_DIR/dw_${vid}.log" 2>&1
    if [ $? -eq 0 ]; then
        local n=$(($(wc -l < "$out") - 1))
        echo "  [winner] v${vid}: DONE ($n labeled)"
    else
        echo "  [winner] v${vid}: FAILED"
    fi
}

# === Phase 1: count_match を並列実行 ===
echo "=== Phase 1: count_match (parallelism=$PARALLELISM) ==="
for vid in "${VIDEOS[@]}"; do
    while [ $(jobs -pr | wc -l) -ge $PARALLELISM ]; do
        sleep 10
    done
    run_count_match "$vid" &
done
wait
echo "=== Phase 1 complete ==="

# === Phase 2: detect_match_winners を並列実行 ===
echo "=== Phase 2: detect_match_winners (parallelism=$PARALLELISM) ==="
ALL_VIDEOS=(04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19)
for vid in "${ALL_VIDEOS[@]}"; do
    while [ $(jobs -pr | wc -l) -ge $PARALLELISM ]; do
        sleep 5
    done
    run_detect_winners "$vid" &
done
wait
echo "=== Phase 2 complete ==="

echo
echo "=== Summary ==="
for vid in "${ALL_VIDEOS[@]}"; do
    matches="data/verify/match_boundaries_v5/video_${vid}/matches.tsv"
    winners="data/verify/match_winners_v${vid}.tsv"
    nm=$([ -s "$matches" ] && echo $(($(wc -l < "$matches") - 1)) || echo "?")
    nw=$([ -s "$winners" ] && echo $(($(wc -l < "$winners") - 1)) || echo "?")
    echo "  v${vid}: matches=$nm winners=$nw"
done
