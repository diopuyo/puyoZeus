#!/usr/bin/env bash
set -u
cd /mnt/d/puyo_analyzer/wt_integ
mkdir -p logs/integration_r1s_dv2d_20261002
export INTEG_NICE=19
bash scripts/run_integration_checks_20261002.sh -m pytest \
  tests/test_exchange_event*.py tests/test_exchange_display_smoothing.py \
  tests/test_exchange_evaluation_cache.py tests/test_e[0-9]*.py \
  tests/test_falsedeath*.py tests/test_post_counter_early_exit.py \
  tests/test_integration_r1s_dv2d.py \
  -k 'not production_off_is_byte_identical' -q -p no:cacheprovider \
  > logs/integration_r1s_dv2d_20261002/pytest.log 2>&1
result=$?
echo "$result" > logs/integration_r1s_dv2d_20261002/pytest.exit
exit "$result"
