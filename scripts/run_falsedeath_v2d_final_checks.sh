#!/bin/bash
# 原票五記録は初回でバイト一致済み。構造整理後の共通回帰を新プロセスで確認する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
bash scripts/run_falsedeath_v2d.sh final_unit -m pytest -q -ra -p no:cacheprovider \
  -k 'not production_off_is_byte_identical' \
  tests/test_falsedeath_v2d.py tests/test_falsedeath_v2.py tests/test_falsedeath_v2c.py \
  tests/test_falsedeath_modes.py tests/test_falsedeath_d1.py tests/test_falsedeath_d2.py \
  tests/test_falsedeath_d3.py tests/test_falsedeath_d4.py tests/test_e35_post_counter_bound.py \
  tests/test_e35b_geometry.py tests/test_e35_integration.py tests/test_post_counter_early_exit.py \
  tests/test_exchange_event_tracker.py tests/test_exchange_event_overlay.py tests/test_exchange_event_production.py
