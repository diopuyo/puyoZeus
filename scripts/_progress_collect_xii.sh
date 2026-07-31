#!/bin/bash
# 30窓再収集の進捗スナップショット (10分ごとの報告用)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
sleep "${1:-600}"
done_n=$(grep -l "rows" logs/collect_xii_v*.log 2>/dev/null | wc -l)
running=$(pgrep -c -f "scripts._collect_1t")
echo "=== progress $(date +%H:%M) ==="
echo "完了ジョブ: ${done_n}/30  実行中プロセス: ${running}"
grep -h "rows" logs/collect_xii_v*.log 2>/dev/null | tail -n 5
tail -n 2 logs/collect_xii_batch6_2026-07-20.log 2>/dev/null
