#!/bin/bash
# 全体採点原票が揃ってから固定条件の比較報告を作る。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2e
while [ ! -f "$OUT/RESULT_v2e.json" ]; do sleep 5; done
bash scripts/run_falsedeath_v2e.sh report -m scripts.falsedeath_v2e_report
date -Iseconds > "$OUT/FINISHED.txt"
