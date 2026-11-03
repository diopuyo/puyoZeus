#!/bin/bash
# カウント除外は確認済みの監視・制御だけに限定する。
set -eu
cd /mnt/d/puyo_analyzer/wt_evalset
source logs/resource_falsedeath.sh
unset FALSEDEATH_CONTROLLER_MODULES
! falsedeath_is_controller 1954
export FALSEDEATH_CONTROLLER_MODULES='scripts.plan_c_t_memory_watch scripts.gfe_phase_stage2_driver_20261002'
falsedeath_is_controller 1954
falsedeath_is_controller 1377
! falsedeath_is_controller 2861
echo 'PASS: 既定は全数、指定時は確認済み制御2本を除外、計算を除外しない'
