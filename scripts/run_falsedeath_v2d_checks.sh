#!/bin/bash
# 共通原票五記録を除外せず、H1と既存死亡経路を検証する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
bash scripts/run_falsedeath_v2d.sh tests -m pytest -q -ra -p no:cacheprovider \
  --basetemp logs/eval_set/falsedeath_fix/v2d/tests_tmp \
  tests/test_falsedeath_v2d.py tests/test_falsedeath_v2.py tests/test_falsedeath_v2c.py \
  tests/test_falsedeath_modes.py tests/test_falsedeath_d1.py tests/test_falsedeath_d2.py \
  tests/test_falsedeath_d3.py tests/test_falsedeath_d4.py tests/test_e35_post_counter_bound.py \
  tests/test_e35b_geometry.py tests/test_e35_integration.py tests/test_post_counter_early_exit.py \
  tests/test_exchange_event_tracker.py tests/test_exchange_event_overlay.py tests/test_exchange_event_production.py
