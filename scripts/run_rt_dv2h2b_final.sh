#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
OUT=logs/rt_dv2h2b_20261003/final
mkdir -p "$OUT"
export INTEG_RESOURCE_OUTPUT=$OUT
# 観測順序・監査保持・worker封印と優先度副作用を最終状態で検証する。
bash scripts/run_rt_integration_checks_20261002.sh -m pytest \
  tests/test_rt_integration_dv2h2b.py tests/test_rt_integration_r1s_dv2d.py \
  tests/test_live_b7.py tests/test_live_b11.py tests/test_live_b16.py \
  tests/test_integration_dv2h2b.py tests/test_e14b_exact_score.py \
  -q -p no:cacheprovider -p scripts.integration_resource_audit > "$OUT/pytest.log" 2>&1
status=$?
echo "$status" > "$OUT/pytest.exit"
exit "$status"
