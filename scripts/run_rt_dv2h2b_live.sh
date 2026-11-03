#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
OUT=$(pwd)/logs/rt_dv2h2b_20261003/live
mkdir -p "$OUT"
export RT_VERIFY_OUTPUT=$OUT/audit
export PUYO_MEMORY_PROFILE=$OUT/memory.jsonl PUYO_MEMORY_TRACE=0 PUYO_MEMORY_INTERVAL=10
# 前回と同じ試合開始を含む71秒。資源条件は共通ランナーで検査する。
bash scripts/run_rt_integration_checks_20261002.sh scripts/verify_rt_dv2h2b_live.py \
  --source video --realtime --start-sec 2579 --end-sec 2650 --warmup-sec 1 \
  --cpu-threads 1 --evaluation-nice 10 --cnn-device cpu --recognition-audit \
  --runtime-audit --output "$OUT" --port 8879 > "$OUT/run.log" 2>&1
status=$?
echo "$status" > "$OUT/exit.txt"
exit "$status"
