#!/bin/bash
# 79動画統合データセット構築パイプライン (2026-08-04 main発注)。
# 拡張13本 (c1,c2,c3,c4,c6,c7,c8,c9,c32,c33,c82,c84,c95) の発火ラベル生成→
# sim付与→既存66動画分とマージ→79動画で三つ巴再学習まで一気通貫。
set -eu
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

EXPAND13_NPZ_DIR="data/indicators_v2/boards_lean_regen_expand13_2026-08-04"
REGEN_NPZ_DIR="data/indicators_v2/boards_lean_regen_2026-07-31"
EXPAND13_VIDEOS="c1 c2 c3 c4 c6 c7 c8 c9 c32 c33 c82 c84 c95"

echo "=== 0. 拡張13本npzを隔離ディレクトリへコピー (既存66本の再計算回避) ==="
mkdir -p "${EXPAND13_NPZ_DIR}"
for v in ${EXPAND13_VIDEOS}; do
  cp "${REGEN_NPZ_DIR}/${v}.npz" "${EXPAND13_NPZ_DIR}/${v}.npz"
done
ls -la "${EXPAND13_NPZ_DIR}"
echo "=== 0完了 $(date) ==="

echo "=== 1. 拡張13本の発火ラベル生成 (--synthesize-terminal-events) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.label_exchange_outcome \
  --npz-dir "${EXPAND13_NPZ_DIR}" --synthesize-terminal-events \
  --output data/indicators_v2/exchange_labels_expand13_2026-08-04.csv
echo "=== 1完了 $(date) ==="

echo "=== 2. 拡張13本のみ sim付与 (--workers 12) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.augment_exchange_labels_with_sim \
  --input-csv data/indicators_v2/exchange_labels_expand13_2026-08-04.csv \
  --npz-dir "${EXPAND13_NPZ_DIR}" \
  --output data/indicators_v2/exchange_labels_expand13_aug_2026-08-04.csv --workers 12
echo "=== 2完了 $(date) ==="

echo "=== 3. 既存66本分+拡張13本分をマージ -> 79動画統合CSV ==="
PYTHONPATH=. ./venv/bin/python -m scripts._merge_79video_labels_2026-08-04
echo "=== 3完了 $(date) ==="

echo "=== 4. 案D 79動画分を再学習 (OOF) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.train_exchange_model_d \
  --labels data/indicators_v2/exchange_labels_regen_synth79_aug_2026-08-04.csv \
  --out-dir data/verify/exchange_model_d_synth79_2026-08-04
echo "=== 4完了 $(date) ==="

echo "=== 5. 併用スタッキング79動画で三つ巴比較 ==="
PYTHONPATH=. ./venv/bin/python -m scripts.run_exchange_triple_comparison \
  --aug-csv data/indicators_v2/exchange_labels_regen_synth79_aug_2026-08-04.csv \
  --model-d-dir data/verify/exchange_model_d_synth79_2026-08-04 \
  --out-dir data/verify/exchange_triple_comparison_synth79_2026-08-04
echo "=== 5完了 $(date) ==="

echo "=== 6. 序盤サンプル数 対比 (66動画版 vs 79動画版) ==="
PYTHONPATH=. ./venv/bin/python -m scripts._report_phase_counts_79_2026-08-04
echo "=== 6完了 $(date) ==="

echo "[all done] $(date)"
