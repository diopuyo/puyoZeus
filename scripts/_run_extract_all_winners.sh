#!/bin/bash
# 全 study 動画 (video_29-38) の試合境界・勝者を抽出する
# 並列 5 本同時実行 (CPU バウンド)
# 使い方: bash scripts/_run_extract_all_winners.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="venv/bin/python"
WINNERS_DIR="data/indicators_v2/winners"
mkdir -p "$WINNERS_DIR"
LOG_DIR="logs/extract_winners"
mkdir -p "$LOG_DIR"

# study 対象動画 (v29-38)
VIDEOS=(29 30 31 32 33 34 35 36 37 38)

echo "[run_extract_all_winners] 開始: $(date)"
echo "対象動画: ${VIDEOS[*]}"
echo ""

pids=()
for v in "${VIDEOS[@]}"; do
    VIDEO="data/frames/video_${v}.mp4"
    OUT_JSON="${WINNERS_DIR}/video_${v}.json"
    LOG="${LOG_DIR}/video_${v}.log"
    if [ ! -f "$VIDEO" ]; then
        echo "[SKIP] 動画なし: $VIDEO"
        continue
    fi
    if [ -f "$OUT_JSON" ]; then
        echo "[SKIP] 既存 JSON あり: $OUT_JSON"
        continue
    fi
    echo "[START] video_${v} -> $OUT_JSON"
    PYTHONPATH=. "$PYTHON" -m scripts.extract_match_winners \
        --video "$VIDEO" \
        --out-json "$OUT_JSON" \
        > "$LOG" 2>&1 &
    pids+=($!)
    # 並列上限 5 本
    if [ ${#pids[@]} -ge 5 ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done

# 残りの完了を待つ
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "[run_extract_all_winners] 完了: $(date)"
echo "生成ファイル:"
ls -la "$WINNERS_DIR/"
