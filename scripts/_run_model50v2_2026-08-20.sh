#!/bin/bash
# 50本 再収集 v2 を切り離し起動する (2026-08-20、境界修正3フラグの配線是正後)。
# 教訓 (memory project_session_2026-08-20_handoff): `setsid -f ./venv/bin/python
# script.py > log` は起動しない。シェルスクリプトを setsid -f bash で起動し、
# スクリプト自身が exec でログへ流し込む方式に統一すること。
#
# 並列数は COLLECT_PARALLEL で上書きできる (既定 14)。物理8コア(論理16)に
# 14プロセスは詰め込みすぎで 1プロセスあたりの実効コアが 0.57 しかないため、
# 落とすと総スループットが上がるかを実測するために可変にした (2026-08-20)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/regen_model50v2_2026-08-20.log 2>&1
echo "=== model50v2 start $(date +%F_%T) parallel=${COLLECT_PARALLEL:-14} ==="
PYTHONPATH=. COLLECT_PARALLEL="${COLLECT_PARALLEL:-14}" ./venv/bin/python -m scripts._regen_model50v2_2026-08-20
echo "=== model50v2 end $(date +%F_%T) rc=$? ==="
