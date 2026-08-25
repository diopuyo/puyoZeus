#!/bin/bash
# 追加14本の収集を切り離し起動する (2026-08-21)。
# 48本 -> 62本 にしてから死に指標の確認に進む (user 指示)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/regen_add14_2026-08-21.log 2>&1
echo "=== add14 start $(date +%F_%T) parallel=${COLLECT_PARALLEL:-14} ==="
PYTHONPATH=. COLLECT_PARALLEL="${COLLECT_PARALLEL:-14}" ./venv/bin/python -m scripts._regen_add14_2026-08-21
echo "=== add14 end $(date +%F_%T) rc=$? ==="
