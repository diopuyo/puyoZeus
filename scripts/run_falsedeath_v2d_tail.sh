#!/bin/bash
# 既存5記録担当の終了後、残りの独立区間を同じ制限・記録単位ロックで再生する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
OUT=logs/eval_set/falsedeath_fix/v2d
test ! -e "$OUT/shard_tail.log"
exec > "$OUT/shard_tail.log" 2>&1
echo "START $(date -Iseconds)"
for source in s4 s5 s6; do
  bash scripts/run_falsedeath_v2d.sh "full_$source" -m scripts.falsedeath_replay_20261001 v2d "$source"
done
echo "END $(date -Iseconds)"
