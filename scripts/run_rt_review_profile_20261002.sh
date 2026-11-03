#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
while [ ! -f logs/rt_review_20261002/with_assets/static_lr/exit.txt ]; do sleep 5; done
export RT_DETAIL_PROFILE=1 RT_STATIC_LR=0
export RT_VERIFY_OUTPUT=$PWD/logs/rt_review_20261002/detail/audit
mkdir -p logs/rt_review_20261002/detail
bash scripts/run_rt_integration_checks_20261002.sh scripts/verify_rt_integration_live_20261002.py --source video --realtime --start-sec 2579 --end-sec 2614 --warmup-sec 1 --cpu-threads 1 --evaluation-nice 10 --cnn-device cpu --recognition-audit --runtime-audit --output logs/rt_review_20261002/detail --port 8879 > logs/rt_review_20261002/detail/run.log 2>&1
