#!/bin/bash
# 変更範囲に関わる既存テスト + 新規テストを nice19・1 プロセスで実行
bash /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed/py.sh -m pytest -q -p no:cacheprovider \
  tests/test_chain.py tests/test_chain_probabilistic.py tests/test_chain_fast_groups.py tests/test_indicators_v2.py \
  tests/test_e23_multilanding.py tests/test_e25_landing_safety.py tests/test_e27_hidden_death.py tests/test_e30_integration.py \
  tests/test_e30_prefire_candidates.py tests/test_e31_prefire_snapshot.py tests/test_e32_hidden_row_belief.py tests/test_e33_stage_timeout.py \
  tests/test_e33b_score_trace.py tests/test_e33b_timeout_only.py tests/test_e34_prefire_origin.py tests/test_e35_integration.py \
  tests/test_e35_post_counter_bound.py tests/test_e35b_geometry.py tests/test_e36_report.py tests/test_d5_single_death_safety.py \
  tests/test_d5b_negative_only.py tests/test_e10_exchange_landing.py tests/test_e10b_exchange_landing.py tests/test_e10c_exchange_landing.py \
  tests/test_e12_exchange_evaluation.py tests/test_e12b_exchange_completion.py tests/test_e20_landing_hands.py tests/test_exchange_event_overlay.py \
  tests/test_exchange_event_overlay_replaced.py tests/test_exchange_event_production.py tests/test_exchange_event_tracker.py \
  tests/test_exchange_fast_expand.py tests/test_multilanding_precheck.py tests/test_exchange_evaluation_cache.py tests/test_post_counter_early_exit.py \
  tests/test_hidden_row_probability.py tests/test_ghost_chain_rule_wiring.py > logs/multilanding_speed/tests_related.log 2>&1
echo "exit=$?" >> logs/multilanding_speed/tests_related.log
