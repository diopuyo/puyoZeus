#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
mkdir -p logs/rt_review_20261002
bash scripts/run_rt_integration_checks_20261002.sh -m pytest tests/phase_j/test_live*.py tests/test_live*.py tests/test_run_live_pipeline_20260928.py tests/test_exchange_event*.py tests/test_exchange_display_smoothing.py tests/test_exchange_evaluation_cache.py tests/test_e[0-9]*.py tests/test_falsedeath*.py tests/test_post_counter_early_exit.py tests/test_integration_r1s_dv2d.py tests/test_rt_integration_r1s_dv2d.py tests/test_rt_review_fixes.py -q -p no:cacheprovider > logs/rt_review_20261002/pytest.log 2>&1
result=$?
echo "$result" > logs/rt_review_20261002/pytest.exit
exit "$result"
