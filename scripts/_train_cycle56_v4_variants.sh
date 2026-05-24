#!/bin/bash
# cycle 56_v4 系: ojama seed 含む 7 クラス真 fine-tune を複数パラメータで連続試行.
# seed 採取完了後に順次起動 (= GPU は 1 つなので並列不可、 順次)。
# 各 candidate は ~5 分で完了見込み。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

SEED_ROOT="data/phase_l/seeds_cycle56_ojama"
BASE_MODEL="models/cnn_phase_b_large_v2.pt"

# 完了確認
if [ ! -d "$SEED_ROOT" ] || [ "$(ls $SEED_ROOT | wc -l)" -lt 25 ]; then
  echo "[fail] ojama seed not ready: $SEED_ROOT"
  exit 1
fi

mkdir -p logs/cycle56_v4_train

# candidate 1: cycle 56_v4 = epochs 2 + lr 1e-5 (= 56_v2 と同設定 + ojama 追加)
echo "=== cycle 56_v4 training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --store-root "$SEED_ROOT" \
  --all \
  --cell-arch large \
  --cell-base-model "$BASE_MODEL" \
  --cell-save-to "models/cnn_cycle56_v4.pt" \
  --epochs 2 \
  --lr 1e-5 \
  --class-balance \
  --augment \
  > logs/cycle56_v4_train/v4.log 2>&1
echo "v4 done @ $(date)"

# candidate 2: cycle 56_v5 = epochs 1 + lr 1e-6 (= さらに軽量、 ojama 重み保持優先)
echo "=== cycle 56_v5 training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --store-root "$SEED_ROOT" \
  --all \
  --cell-arch large \
  --cell-base-model "$BASE_MODEL" \
  --cell-save-to "models/cnn_cycle56_v5.pt" \
  --epochs 1 \
  --lr 1e-6 \
  --class-balance \
  --augment \
  > logs/cycle56_v4_train/v5.log 2>&1
echo "v5 done @ $(date)"

# candidate 3: cycle 56_v6 = epochs 3 + lr 5e-6 + focal-gamma 2.0 (= ojama 重視)
echo "=== cycle 56_v6 training @ $(date) ==="
PYTHONPATH=. ./venv/bin/python -m scripts.phase_i_fine_tune \
  --component cell_color \
  --store-root "$SEED_ROOT" \
  --all \
  --cell-arch large \
  --cell-base-model "$BASE_MODEL" \
  --cell-save-to "models/cnn_cycle56_v6.pt" \
  --epochs 3 \
  --lr 5e-6 \
  --class-balance \
  --focal-gamma 2.0 \
  --augment \
  > logs/cycle56_v4_train/v6.log 2>&1
echo "v6 done @ $(date)"

echo "=== all candidates trained @ $(date) ==="
