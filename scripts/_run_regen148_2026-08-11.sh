#!/bin/bash
# 148動画 学習データ再生成パイプライン ランナー (2026-08-11、user承認済み)。
# scripts/_regen148_orchestrator_2026-08-11.py の起動シェル (CYCLE_FINDINGS
# §3.4 に倣い、引用符ネスト事故を避けるためスクリプトファイル経由で detach する)。
#
# 使い方 (WSL detach、長時間放置前提、CLAUDE.md プロセス管理ルール):
#   wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#     setsid -f bash scripts/_run_regen148_2026-08-11.sh < /dev/null"
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
# 1プロセス=1コアを厳守 (matchTemplate 系スレッド競合防止、
# memory `project_collect_indicators_v2_perf_2026-07-20` 教訓)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1
mkdir -p logs
LOG="logs/regen_2026-08-11.log"
echo "[run_regen148] 開始 $(date)" >> "${LOG}"
./venv/bin/python -u scripts/_regen148_orchestrator_2026-08-11.py >> "${LOG}" 2>&1
echo "[run_regen148] スクリプト終了 $(date) rc=$?" >> "${LOG}"
