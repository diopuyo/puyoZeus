#!/bin/bash
# W7+W8: pl5/pl6 全 16 動画から
#   (1) 代表 1 試合の field overlay 動画 (動画レビュー)
#   (2) uncertain パッチ抽出 (画像レビュー)
# を生成。

set -e
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

PY="./venv/bin/python"
OUT_VIDEO_DIR="data/verify/phase_w_review/videos"
OUT_UNCERTAIN_DIR="data/verify/phase_w_review/uncertain"
mkdir -p "$OUT_VIDEO_DIR" "$OUT_UNCERTAIN_DIR"

VIDEOS="04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19"

# 各動画で 60-80 秒の代表試合を 1 つ選ぶ + 抽出時刻 (試合内 4 時刻) を生成
pick_match() {
    local vid=$1
    local matches="data/verify/match_winners_v${vid}.tsv"
    if [ ! -s "$matches" ]; then
        echo ""
        return
    fi
    # 60-80 秒の試合を抽出して先頭 1 つ
    awk -F'\t' 'NR>1 && ($3-$2) >= 60 && ($3-$2) <= 80 {print $2"\t"$3"\t"$4; exit}' "$matches"
}

# (1) Field overlay 動画生成
render_field_video() {
    local vid=$1
    local start=$2
    local end=$3
    local out="$OUT_VIDEO_DIR/v${vid}_field.mp4"
    if [ -s "$out" ]; then
        echo "  v${vid} field: SKIP"
        return
    fi
    local dur=$(awk "BEGIN{print $end - $start}")
    echo "  v${vid} field: render $start..$end ($dur s)"
    PYTHONPATH=. $PY -m scripts.phase_u_render \
        "data/frames/video_${vid}.mp4" "$out" \
        --interval 0.2 --start-sec $start --max-seconds $dur \
        > "/tmp/v${vid}_field.log" 2>&1 || echo "    FAILED"
}

# (2) Uncertain パッチ抽出
extract_uncertain() {
    local vid=$1
    local start=$2
    local end=$3
    local out_dir="$OUT_UNCERTAIN_DIR/v${vid}"
    if [ -s "$out_dir/labels.csv" ]; then
        echo "  v${vid} uncertain: SKIP"
        return
    fi
    # 試合内で 4 時刻に分散
    local mid=$(awk "BEGIN{print ($start+$end)/2}")
    local t1=$(awk "BEGIN{printf \"%.1f\", $start + ($end-$start)*0.30}")
    local t2=$(awk "BEGIN{printf \"%.1f\", $start + ($end-$start)*0.50}")
    local t3=$(awk "BEGIN{printf \"%.1f\", $start + ($end-$start)*0.70}")
    local t4=$(awk "BEGIN{printf \"%.1f\", $start + ($end-$start)*0.85}")
    echo "  v${vid} uncertain: extract t=$t1,$t2,$t3,$t4"
    PYTHONPATH=. $PY -m scripts.phase_u_extract_uncertain \
        "data/frames/video_${vid}.mp4" \
        --times "$t1,$t2,$t3,$t4" \
        --out-dir "$out_dir" \
        --cnn-model models/cnn_phase_u_v7.pt \
        --threshold 0.80 --max-samples 60 \
        > "/tmp/v${vid}_uncertain.log" 2>&1 || echo "    FAILED"
}

echo "=== Phase 1: pick matches ==="
for vid in $VIDEOS; do
    info=$(pick_match $vid)
    if [ -z "$info" ]; then
        echo "  v${vid}: no 60-80s match"
        continue
    fi
    start=$(echo "$info" | cut -f1)
    end=$(echo "$info" | cut -f2)
    winner=$(echo "$info" | cut -f3)
    echo "  v${vid}: $start..$end ($winner)"
done

echo ""
echo "=== Phase 2: field overlay videos (sequential) ==="
for vid in $VIDEOS; do
    info=$(pick_match $vid)
    [ -z "$info" ] && continue
    start=$(echo "$info" | cut -f1)
    end=$(echo "$info" | cut -f2)
    render_field_video $vid $start $end
done

echo ""
echo "=== Phase 3: uncertain patches (sequential) ==="
for vid in $VIDEOS; do
    info=$(pick_match $vid)
    [ -z "$info" ] && continue
    start=$(echo "$info" | cut -f1)
    end=$(echo "$info" | cut -f2)
    extract_uncertain $vid $start $end
done

echo ""
echo "=== complete ==="
ls "$OUT_VIDEO_DIR/"
ls "$OUT_UNCERTAIN_DIR/"
