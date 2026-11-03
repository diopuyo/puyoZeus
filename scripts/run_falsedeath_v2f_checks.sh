#!/bin/bash
# 周辺回帰と既定OFFの本番原票一致を、同じ資源制限で確認する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
bash scripts/run_falsedeath_v2f.sh regression_tests -m pytest \
  tests/test_falsedeath*.py tests/test_exchange_event*.py \
  tests/test_exchange_display_smoothing.py tests/test_post_counter_early_exit.py \
  -q -p no:cacheprovider --basetemp logs/eval_set/falsedeath_fix/v2f/tests_tmp
