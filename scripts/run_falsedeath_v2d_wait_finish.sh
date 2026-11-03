#!/bin/bash
# 全体採点の完了印を待ち、検算・全表化を一度だけ起動する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2d
test ! -e "$OUT/finish_worker.log"
exec > "$OUT/finish_worker.log" 2>&1
echo "WAIT $(date -Iseconds)"
while ! grep -q '^FULL_SCORED ' "$OUT/worker.log" && ! grep -q '^STOP_LADDER_FAILED ' "$OUT/worker.log"; do
  sleep 10
done
echo "START $(date -Iseconds)"
bash scripts/run_falsedeath_v2d_finish.sh
echo "END $(date -Iseconds)"
