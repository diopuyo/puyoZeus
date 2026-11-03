#!/bin/bash
# 実処理時間を分離した検算から再開し、既に完成した採点・集計は上書きしない。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2d
test ! -e "$OUT/finish_remaining.log"
exec > "$OUT/finish_remaining.log" 2>&1
echo "START $(date -Iseconds)"
bash scripts/run_falsedeath_v2d.sh verify2 -m scripts.falsedeath_v2d_verify
bash scripts/run_falsedeath_v2d.sh receipt -m scripts.falsedeath_v2d_receipt
bash scripts/run_falsedeath_v2d.sh report -m scripts.falsedeath_v2d_report
echo "END $(date -Iseconds)"
