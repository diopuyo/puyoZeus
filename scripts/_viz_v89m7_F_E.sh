#!/bin/bash
# v89m7 を F + E 適用済 state machine で viz 生成 (= ユーザー目視用).
# 現 default cnn_phase_b_large_v2.pt + F (STABLE 復帰ゲート) + E (HSV 赤色循環バグ修正)
# の効果をユーザーが直接目視確認するための viz。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health viz_v89m7_F_E

mkdir -p data/verify/cycle56_viz
mkdir -p logs/cycle56_viz

INPUT="data/phase_l/cut/v89m7_buf15s.mp4"
CNN_MODEL="models/cnn_phase_b_large_v2.pt"
OUTPUT="data/verify/cycle56_viz/v89m7_F_E.mp4"
BOARD_LOG="logs/cycle56_viz/v89m7_F_E_board.jsonl"

if [ ! -f "$INPUT" ]; then
  echo "[fail] input not found: $INPUT"
  finalize_health 1
  exit 1
fi
if [ ! -f "$CNN_MODEL" ]; then
  echo "[fail] CNN model not found: $CNN_MODEL"
  finalize_health 1
  exit 1
fi

echo "=== v89m7 F + E viz @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.visualize_recognition \
  --video "$INPUT" \
  --output "$OUTPUT" \
  --cnn-model "$CNN_MODEL" \
  --hsv-state data/per_video_hsv_ranges/_merged_default.json \
  --dump-board-log "$BOARD_LOG" \
  > logs/cycle56_viz/v89m7_F_E.log 2>&1

if [ ! -f "$OUTPUT" ]; then
  echo "[fail] viz output not generated"
  finalize_health 1
  exit 1
fi

echo "[done] viz -> $OUTPUT"
echo "[done] board_log -> $BOARD_LOG"

# 改善前 (= 2026-05-21 14:50-16:04 生成済) との比較情報
echo ""
echo "=== 比較対象 (= F + E 適用前 baseline) ==="
echo "baseline mp4: data/verify/cycle55_viz/v89m7_ojama_baseline.mp4 (= 16:04 生成、 291MB)"
echo "F + E mp4   : $OUTPUT"
echo ""
echo "ユーザー目視ポイント (= 4 軸):"
echo "  1. 背景→ぷよ 2 秒残るパターン: F 効果で削減されているか"
echo "  2. 赤↔黄 / 赤↔青 誤認: E 効果で削減されているか"
echo "  3. ojama 認識: baseline と同等 OK か (= 退行なし)"
echo "  4. その他誤認: 新規回帰がないか"

finalize_health 0
