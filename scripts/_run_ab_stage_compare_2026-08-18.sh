#!/bin/bash
# 148収集の進捗断面 (30/60/100/144本) ごとの新旧構成A/B比較 ランナー (2026-08-18)。
# scripts/_ab_stage_compare_2026-08-18.py の起動シェル (CYCLE_FINDINGS §3.4に
# 倣い、引用符ネスト事故を避けるためスクリプトファイル経由で detach する)。
#
# 使い方 (WSL detach、長時間放置前提、CLAUDE.md プロセス管理ルール):
#   wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#     setsid -f bash scripts/_run_ab_stage_compare_2026-08-18.sh < /dev/null"
#
# 148再収集 (14並列) + 別途走行予定の動画解析ジョブとCPUを奪い合う前提のため、
# このスクリプト自体もプロセス全体を nice -n 19 に落とす (子プロセスへも継承される)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
mkdir -p logs
LOG="logs/ab_stage_compare_2026-08-18.log"
echo "[run_ab_stage_compare] 開始 $(date)" >> "${LOG}"
nice -n 19 ./venv/bin/python -u scripts/_ab_stage_compare_2026-08-18.py >> "${LOG}" 2>&1
echo "[run_ab_stage_compare] スクリプト終了 $(date) rc=$?" >> "${LOG}"
