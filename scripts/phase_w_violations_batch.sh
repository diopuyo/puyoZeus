#!/bin/bash
# W8: pl5/pl6 全 16 動画から物理矛盾セル (4+ 連結 + 隣接) を抽出。
# 各動画で代表 1 試合区間に対して実行。

set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
OUT_BASE="data/verify/phase_w_review/violations_50_bg"
mkdir -p "$OUT_BASE"
LOG_DIR="/tmp/violations_batch_50_bg"
mkdir -p "$LOG_DIR"

VIDEOS="04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19"

extract_for_video() {
    local vid=$1
    local matches="data/verify/match_winners_v${vid}.tsv"
    if [ ! -s "$matches" ]; then
        return
    fi
    # 60-80 秒の代表試合
    local info=$(awk -F'\t' 'NR>1 && ($3-$2) >= 60 && ($3-$2) <= 80 {print $2"\t"$3; exit}' "$matches")
    if [ -z "$info" ]; then
        info=$(awk -F'\t' 'NR>1 && ($3-$2) >= 50 && ($3-$2) <= 100 {print $2"\t"$3; exit}' "$matches")
    fi
    if [ -z "$info" ]; then
        return
    fi
    local start=$(echo "$info" | cut -f1)
    local end=$(echo "$info" | cut -f2)
    local out_dir="$OUT_BASE/v${vid}"
    if [ -s "$out_dir/labels.csv" ]; then
        echo "  v${vid}: SKIP"
        return
    fi
    echo "  v${vid}: $start..$end"
    # BG FP 用の試合開始秒 (start - 2 秒、開始準備画面で安定背景)
    local bg_t=$(awk "BEGIN{printf \"%.1f\", $start - 2}")
    PYTHONPATH=. $PY -m scripts.phase_w_extract_violations \
        "data/frames/video_${vid}.mp4" \
        --start $start --end $end \
        --interval 5.0 --max-samples 50 \
        --bg-fp-time $bg_t \
        --out-dir "$out_dir" \
        > "$LOG_DIR/v${vid}.log" 2>&1
    if [ -s "$out_dir/labels.csv" ]; then
        local n=$(($(wc -l < "$out_dir/labels.csv") - 1))
        echo "  v${vid}: $n violation cells"
    else
        echo "  v${vid}: 0 violations"
    fi
}

echo "=== violation extraction (sequential, max 200/video) ==="
for vid in $VIDEOS; do
    extract_for_video $vid
done

echo ""
echo "=== complete ==="
total=0
for vid in $VIDEOS; do
    csv="$OUT_BASE/v${vid}/labels.csv"
    if [ -s "$csv" ]; then
        n=$(($(wc -l < "$csv") - 1))
        total=$((total + n))
    fi
done
echo "total: $total cells across $(ls $OUT_BASE 2>/dev/null | wc -l) videos"
