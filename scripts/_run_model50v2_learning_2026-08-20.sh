#!/bin/bash
# 48本収集後の学習工程を一括実行する (2026-08-20)。
#
# docs/RECOLLECT148_2026-08-18.md の「収集後の工程 (実行順・確定版)」に従う。
# 手打ちを避けてスクリプト化する理由: --exclude-match-end-locked は
# LEARNING_DATA_BUILD_ADOPTED に採用登録済みだが**既定 False で付け忘れても
# 何も警告されない**と手順書に明記されている。今日だけで配線漏れ事故を
# 3件踏んでいるため (memory feedback_wiring_check_needs_nongeneric_scripts_
# 2026-08-18)、フラグをスクリプトに埋め込んで固定する。
#
# 所要の見積もり (手順書の実測値を本数比で換算):
#   品質ゲート  数分
#   CSVビルド   148本で4.4時間 -> 48本で約1.5時間
#   再学習      約32分
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/model50v2_learning_2026-08-20.log 2>&1

NPZ_DIR=data/indicators_v2/boards_lean_model50v2_2026-08-20
CSV_DIR=data/verify/labeled_win_model62_2026-08-21
OUT_DIR=data/verify/retrain_model62_2026-08-21
GATE_DIR=data/verify/quality_gate_model62_2026-08-21

echo "=== 学習工程 start $(date +%F_%T) ==="
N=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | wc -l)
echo "[入力] npz $N 本"
if [ "$N" -lt 62 ]; then
  echo "[中止] 62本揃っていない ($N 本)。収集完了を待つこと。"
  echo "  欠測のまま学習すると前回結果と比較できなくなる (手順書の警告)"
  exit 1
fi

echo "--- 1. 品質ゲート $(date +%T) ---"
PYTHONPATH=. ./venv/bin/python scripts/phase_l_video_quality_gate.py --all \
    --npz-dir "$NPZ_DIR" --out-dir "$GATE_DIR"
echo "  rc=$?"

echo "--- 2. 学習CSVビルド $(date +%T) ---"
mkdir -p "$CSV_DIR"
PYTHONPATH=. ./venv/bin/python -m scripts.build_labeled_win_from_npz \
    --npz-dir "$NPZ_DIR" \
    --out "$CSV_DIR/labeled_win_model62.csv" \
    --profile full \
    --exclude-match-end-locked
echo "  rc=$?"

echo "--- 3. 再学習・評価 $(date +%T) ---"
PYTHONPATH=. ./venv/bin/python scripts/_retrain148_2026-08-14.py \
    --csv "$CSV_DIR/labeled_win_model62.csv" \
    --out-dir "$OUT_DIR"
echo "  rc=$?"

echo "=== 学習工程 end $(date +%F_%T) ==="
