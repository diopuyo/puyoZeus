#!/bin/bash
# フェーズ3(label_exchange_outcome on boards_lean_next) + フェーズ4(proto_net_threat_v2 --full)
# を25本npz収集完了後に一括実行する。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3

echo "=== 収集本数確認 ==="
n=$(ls data/indicators_v2/boards_lean_next/*.npz 2>/dev/null | wc -l)
echo "boards_lean_next npz数: ${n}"

echo ""
echo "=== フェーズ3: label_exchange_outcome (boards_lean_next入力) ==="
./venv/bin/python -m scripts.label_exchange_outcome \
    --npz-dir data/indicators_v2/boards_lean_next \
    --output data/indicators_v2/exchange_labels_next.csv

echo ""
echo "=== フェーズ4: proto_net_threat_v2 --full ==="
./venv/bin/python -m scripts.proto_net_threat_v2 --full

echo ""
echo "[phase34] ALL DONE"
