#!/bin/bash
# W8: pl5/pl6 全 16 動画から uncertain パッチを抽出してレビューシート生成。
# 動画レビュー省略、画像 (シート) レビューに集中。

set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
OUT_BASE="data/verify/phase_w_review/uncertain"
mkdir -p "$OUT_BASE"
LOG_DIR="/tmp/uncertain_batch"
mkdir -p "$LOG_DIR"

VIDEOS="04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19"

# 各動画で 60-80 秒の代表試合を 1 つ取得 + 4 時刻
extract_for_video() {
    local vid=$1
    local matches="data/verify/match_winners_v${vid}.tsv"
    if [ ! -s "$matches" ]; then
        echo "  v${vid}: no winners.tsv, SKIP"
        return
    fi
    # 60-80 秒の試合
    local info=$(awk -F'\t' 'NR>1 && ($3-$2) >= 60 && ($3-$2) <= 80 {print $2"\t"$3; exit}' "$matches")
    if [ -z "$info" ]; then
        # 50-100 秒に緩和
        info=$(awk -F'\t' 'NR>1 && ($3-$2) >= 50 && ($3-$2) <= 100 {print $2"\t"$3; exit}' "$matches")
    fi
    if [ -z "$info" ]; then
        echo "  v${vid}: no suitable match"
        return
    fi
    local start=$(echo "$info" | cut -f1)
    local end=$(echo "$info" | cut -f2)
    local out_dir="$OUT_BASE/v${vid}"
    if [ -s "$out_dir/labels.csv" ]; then
        echo "  v${vid}: SKIP (already done)"
        return
    fi
    # 試合内 4 時刻
    local t1=$(awk "BEGIN{printf \"%.1f\", $start + ($end-$start)*0.25}")
    local t2=$(awk "BEGIN{printf \"%.1f\", $start + ($end-$start)*0.45}")
    local t3=$(awk "BEGIN{printf \"%.1f\", $start + ($end-$start)*0.65}")
    local t4=$(awk "BEGIN{printf \"%.1f\", $start + ($end-$start)*0.85}")
    echo "  v${vid}: $start..$end, t=$t1,$t2,$t3,$t4"
    PYTHONPATH=. $PY -m scripts.phase_u_extract_uncertain \
        "data/frames/video_${vid}.mp4" \
        --times "$t1,$t2,$t3,$t4" \
        --out-dir "$out_dir" \
        --cnn-model models/cnn_phase_u_v7.pt \
        --threshold 0.80 --max-samples 100 \
        > "$LOG_DIR/v${vid}.log" 2>&1
    if [ -s "$out_dir/labels.csv" ]; then
        local n=$(($(wc -l < "$out_dir/labels.csv") - 1))
        echo "  v${vid}: $n cells extracted"
    else
        echo "  v${vid}: FAILED (see $LOG_DIR/v${vid}.log)"
    fi
}

echo "=== uncertain extraction (sequential, max 100 cells/video) ==="
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
        echo "  v${vid}: $n cells, sheet: $OUT_BASE/v${vid}/sheet.png"
    fi
done
echo ""
echo "total: $total cells"
