#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
OUT=${INTEG_OUTPUT:-logs/integration_dv2h2b_20261003}
# A/Bで同じ関連回帰を使い、rt側だけlive一式を追加する。
tests=(tests/test_exchange_event*.py tests/test_exchange_display_smoothing.py
  tests/test_exchange_evaluation_cache.py tests/test_e[0-9]*.py
  tests/test_falsedeath*.py tests/test_post_counter_early_exit.py
  tests/test_integration*.py)
runner=scripts/run_integration_checks_20261002.sh
if [ -f src/phase_j/live_evaluation.py ]; then
  OUT=${INTEG_OUTPUT:-logs/rt_dv2h2b_20261003}
  runner=scripts/run_rt_integration_checks_20261002.sh
  tests+=(tests/test_live*.py tests/phase_j/test_live*.py tests/test_rt_integration*.py)
fi
mkdir -p "$OUT"
export INTEG_NICE=10
export INTEG_RESOURCE_OUTPUT=$OUT
bash "$runner" -m pytest "${tests[@]}" -k 'not production_off_is_byte_identical' \
  -q -p no:cacheprovider -p scripts.integration_resource_audit > "$OUT/pytest.log" 2>&1
status=$?
echo "$status" > "$OUT/pytest.exit"
exit "$status"
