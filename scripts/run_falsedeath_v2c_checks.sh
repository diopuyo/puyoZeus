#!/bin/bash
# C6を除外せず、同じ資源制限で原票5記録を含む共通テストを実行する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
bash scripts/run_falsedeath_v2c.sh tests -m pytest -q -ra -p no:cacheprovider \
  --basetemp logs/eval_set/falsedeath_fix/v2c/tests_tmp \
  tests/test_falsedeath_v2.py tests/test_falsedeath_v2c.py tests/test_falsedeath_modes.py \
  tests/test_falsedeath_d1.py tests/test_falsedeath_d2.py tests/test_falsedeath_d3.py tests/test_falsedeath_d4.py \
  tests/test_e35_post_counter_bound.py tests/test_e35b_geometry.py tests/test_e35_integration.py \
  tests/test_post_counter_early_exit.py tests/test_exchange_event_tracker.py \
  tests/test_exchange_event_overlay.py tests/test_exchange_event_production.py
