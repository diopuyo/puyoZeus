#!/bin/bash
# #24 Step0 前提ゲート再検証 (2026-07-28): 22動画バッチ + c5テストジョブの
# 完走をポーリング待機し、完走後に自動で再測定+summary.md生成を実行する。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONPATH=.

echo "[wait] $(date +%s) 完走待機開始"
while pgrep -f "collect_lean_1t|collect_boards_lean" > /dev/null; do
  sleep 60
done
echo "[wait] $(date +%s) 全ジョブ完走を検知"

nice -n 15 ./venv/bin/python -m scripts._finalize_gate_recheck_2026-07-28 \
  > logs/finalize_gate_recheck_2026-07-28.log 2>&1
echo "[finalize] $(date +%s) 完了 (exit=$?)" >> logs/finalize_gate_recheck_2026-07-28.log
