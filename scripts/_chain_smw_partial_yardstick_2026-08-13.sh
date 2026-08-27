#!/bin/bash
# 多数決窓の部分物差し検証チェーン (2026-08-13):
# OFF収集完了待ち → ON収集 → 両方の物差し測定 → 結果をログ末尾に出力
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
# OFF側の完了待ち (最大4時間)
for i in $(seq 1 480); do
  grep -q "ALL DONE" logs/smw_off_2026-08-13.log 2>/dev/null && break
  sleep 30
done
if ! grep -q "ALL DONE" logs/smw_off_2026-08-13.log 2>/dev/null; then
  echo "SMW_CHAIN_TIMEOUT_OFF_COLLECT"; exit 1
fi
bash scripts/_run_smw_on_2026-08-13.sh
if ! grep -q "ALL DONE" logs/smw_on_2026-08-13.log 2>/dev/null; then
  echo "SMW_CHAIN_ON_COLLECT_INCOMPLETE"; exit 1
fi
echo "=== 物差し OFF (現行) ==="
./venv/bin/python -m scripts._measure_yardstick_v4_2026-08-05 --v4-npz-dir data/verify/board_labels_smw_off_2026-08-13
echo "=== 物差し ON (多数決) ==="
./venv/bin/python -m scripts._measure_yardstick_v4_2026-08-05 --v4-npz-dir data/verify/board_labels_smw_on_2026-08-13
echo "SMW_PARTIAL_YARDSTICK_DONE"
