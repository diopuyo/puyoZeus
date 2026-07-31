#!/bin/bash
# 収集完了を待って自動的にフェーズ3+4まで走らせる自律パイプライン。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
LOG=logs/lean_next_rest23_2026-07-21.log
MAX_WAIT_SEC=7200
ELAPSED=0
INTERVAL=20
while [ "$ELAPSED" -lt "$MAX_WAIT_SEC" ]; do
    if grep -q "ALL DONE" "$LOG" 2>/dev/null; then
        echo "[autopipeline] 収集ALL DONE検知 (elapsed=${ELAPSED}s)、フェーズ3+4開始"
        bash scripts/_run_phase34_2026-07-21.sh
        exit $?
    fi
    n=$(ls data/indicators_v2/boards_lean_next/*.npz 2>/dev/null | wc -l)
    echo "[autopipeline] elapsed=${ELAPSED}s 完了npz数=${n}/25"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done
echo "[autopipeline] タイムアウト (${MAX_WAIT_SEC}s、収集未完了)"
exit 1
