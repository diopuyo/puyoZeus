#!/bin/bash
# 残り23本収集の完了を待つ(ALL DONEログ検知 or タイムアウト)。
# fd/途中npzは進捗指標にしない (memory project_collect_indicators_v2_perf_2026-07-20 の教訓)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
LOG=logs/lean_next_rest23_2026-07-21.log
MAX_WAIT_SEC=7200   # 最大2時間待つ (安全上限)
ELAPSED=0
INTERVAL=30
while [ "$ELAPSED" -lt "$MAX_WAIT_SEC" ]; do
    if grep -q "ALL DONE" "$LOG" 2>/dev/null; then
        echo "[wait] ALL DONE検知 (elapsed=${ELAPSED}s)"
        exit 0
    fi
    n=$(ls data/indicators_v2/boards_lean_next/*.npz 2>/dev/null | wc -l)
    echo "[wait] elapsed=${ELAPSED}s 完了npz数=${n}/25"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done
echo "[wait] タイムアウト (${MAX_WAIT_SEC}s経過、未完了の可能性)"
exit 1
