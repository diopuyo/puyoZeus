#!/usr/bin/env bash
# Phase H4.2 学習ランチャー (Stage 3).
#
# 前提:
#   1. scripts/phase_h2_collect_board.py が完了済み
#      (data/training/phase_h2_boards/v??.npz が 11 件以上)
#   2. data/training/match_features_phase_h2_quick_with_board.csv が存在
#
# 実行:
#   ./scripts/_run_phase_h4_2_train.sh
#
# 出力:
#   logs/phase_h4_2_train.log
#   data/verify/phase_h4_2_results.json
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BOARD_DIR="data/training/phase_h2_boards"
CSV="data/training/match_features_phase_h2_quick_with_board.csv"
OUT="data/verify/phase_h4_2_results.json"
LOG="logs/phase_h4_2_train.log"

# 前提チェック
if [ ! -d "$BOARD_DIR" ]; then
    echo "[error] board NPZ ディレクトリがない: $BOARD_DIR" >&2
    exit 1
fi
n_npz=$(find "$BOARD_DIR" -maxdepth 1 -name 'v??.npz' | wc -l)
if [ "$n_npz" -lt 1 ]; then
    echo "[error] board NPZ が見つからない: $BOARD_DIR" >&2
    exit 1
fi
if [ ! -f "$CSV" ]; then
    echo "[error] CSV がない: $CSV" >&2
    exit 1
fi

mkdir -p logs data/verify

echo "[start] phase_h4_2_train (n_npz=$n_npz)"
PYTHONPATH=. ./venv/bin/python -u -m scripts.phase_h4_2_train \
    --board-dir "$BOARD_DIR" \
    --csv "$CSV" \
    --out "$OUT" \
    --log "$LOG" \
    --device auto
echo "[done] -> $OUT"
