#!/bin/bash
# 完成済み記録を飛ばし、資源制限の管理専用プロセス識別を更新して再開する。
set -eu
cd /mnt/d/puyo_analyzer/wt_a1
export A1_MODES=$1
bash scripts/run_a1_completion.sh
