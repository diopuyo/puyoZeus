#!/bin/bash
# 画像窓の確定を待つ間に、短区間の全JSON行を本収集の書出し済み先頭と照合する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python -B - <<'PY'
import json
from pathlib import Path
from scripts.eval_set_prefix_check_20261001 import compare_rows

root = Path('logs/eval_set/set2')
result = compare_rows(root/'collect/records/s1.jsonl.tmp', root/'collect/records/check.jsonl.gz')
(root/'PREFIX_ROWS_EARLY.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
print(json.dumps(result, ensure_ascii=False), flush=True)
assert result['mismatched_rows'] == 0 and result['check_rows_unpaired'] == 0
PY
