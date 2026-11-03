#!/bin/bash
# 変更周辺の回帰と実原票の既定OFF互換性を、独立した資源制限ジョブで確認する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
bash scripts/run_falsedeath_v2e.sh regression_tests -m pytest \
  tests/test_falsedeath*.py tests/test_exchange_event*.py \
  tests/test_exchange_display_smoothing.py tests/test_post_counter_early_exit.py \
  -q -p no:cacheprovider --basetemp logs/eval_set/falsedeath_fix/v2e/tests_tmp
while [ ! -f logs/eval_set/falsedeath_fix/v2e/default_off/status.json ]; do sleep 5; done
if [ ! -f logs/eval_set/falsedeath_fix/v2e/VERIFICATION.json ]; then
  bash scripts/run_falsedeath_v2e.sh verification -m scripts.verify_falsedeath_v2e
fi
