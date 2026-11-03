#!/bin/bash
# c13ea95のソースをそのまま読み、既存5記録の本番互換を確認する。
set -eu
cd /mnt/d/puyo_analyzer/wt_a2
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export FALSEDEATH_POLL_SECONDS="${FALSEDEATH_POLL_SECONDS:-2}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT="$PWD/logs/a2_safety"
source scripts/a2_resource_limits.sh
exec 8> "$OUT/reference.lock"
flock 8
PY=<home>/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
for name in review q_7gc4TgFig fcXG83vInDY mia8KCjr52g zenchi; do
  dest="$OUT/replay/reference/$name"
  if [ -f "$dest/status.json" ]; then continue; fi
  run_limited "$OUT/reference_$name.log" nice -n 10 "$PY" -B -m scripts.a2_reference \
    "/mnt/d/puyo_analyzer/wt_evalset/logs/eval_set/falsedeath_fix/collect/records/$name.jsonl.gz" \
    --out "$dest" --production-exchange-event
done
