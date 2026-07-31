#!/bin/bash
# 同期ポーリング用: レンダ完走を60秒間隔・9回(=9分)で確認する。
# 呼出元が10分以内のBash呼び出しを繰り返し発行する想定 (feedback_msys_pipe_escape.md)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
LOG="logs/review_51flags_video84_2026-07-27.log"
for i in $(seq 1 9); do
  if grep -q "\[all done\]" "${LOG}" 2>/dev/null; then
    echo "FOUND_DONE"
    break
  fi
  echo "[poll ${i}] $(date +%T)"
  tail -3 "${LOG}"
  sleep 60
done
echo "LOOP_END"
