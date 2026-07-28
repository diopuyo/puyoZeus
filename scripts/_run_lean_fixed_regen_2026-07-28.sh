#!/bin/bash
# #24 Step0 前提ゲート再検証 (2026-07-28): 認識強化後の設定で c 系 22 動画
# (c5 はテスト実行済のため除く) を boards_lean_fixed_regen_2026-07-28/ に
# 再収集する。並列4本・nice 15 (m30 収集との競合回避、userタスク指定)。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p data/indicators_v2/boards_lean_fixed_regen_2026-07-28 logs
export PYTHONPATH=.
JOBLOG=logs/lean_fixed_regen_2026-07-28_joblog.txt
xargs -a scripts/_jobs_lean_fixed_regen_2026-07-28.txt -d '\n' -P 4 -I CMD \
  bash -c 'nice -n 15 CMD' \
  > logs/lean_fixed_regen_2026-07-28_stdout.log 2>&1
echo "ALL_DONE $(date +%s)" >> logs/lean_fixed_regen_2026-07-28_stdout.log
