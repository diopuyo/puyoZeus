#!/bin/bash
# 配線是正の2本検証収集を切り離し起動する (2026-08-20)。
# 教訓 (memory project_session_2026-08-20_handoff): `setsid -f ./venv/bin/python
# script.py > log` は起動しない。シェルスクリプトを setsid -f bash で起動し、
# スクリプト自身が exec でログへ流し込む方式に統一すること。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/regen_wirecheck_2026-08-20.log 2>&1
echo "=== wirecheck2 start $(date +%F_%T) ==="
PYTHONPATH=. ./venv/bin/python -m scripts._regen_wirecheck2_2026-08-20
echo "=== wirecheck2 end $(date +%F_%T) rc=$? ==="
