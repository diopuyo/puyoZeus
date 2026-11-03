#!/bin/bash
# 互換再生後に本番5再生、18記録の採点後に全表と受領票を生成する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2h2b
exec > "$OUT/finish_worker.log" 2>&1
while [ ! -f "$OUT/compat_v2h2/review/COMPATIBILITY.json" ]; do sleep 5; done
bash scripts/run_falsedeath_v2h2b.sh production_tests -m pytest tests/test_falsedeath_v2.py -q -k test_production_off_is_byte_identical -p no:cacheprovider --basetemp "$OUT/production_tmp"
while [ ! -f "$OUT/RESULT_v2h2b.json" ]; do sleep 5; done
bash scripts/run_falsedeath_v2h2b.sh coverage -m scripts.falsedeath_v2h2b_coverage
bash scripts/run_falsedeath_v2h2b.sh report -m scripts.falsedeath_v2h2b_report
bash scripts/run_falsedeath_v2h2b.sh freshness -m scripts.falsedeath_v2h2b_freshness
bash scripts/run_falsedeath_v2h2b.sh receipt -m scripts.falsedeath_v2h2b_receipt
echo "FINISHED $(date -Iseconds)"
