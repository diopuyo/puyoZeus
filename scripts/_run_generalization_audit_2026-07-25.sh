#!/bin/bash
# 汎化監査(2026-07-25) 起動スクリプト。
# 調整に使っていない8動画 x (新既定/旧既定) = 16ジョブを scripts/_run_safe.sh
# (熱対策: MAXPAR=3, THREADS=3, COOLDOWN=60) で実行し、完走後に
# scripts/_aggregate_generalization_audit_2026-07-25.py を自動実行して
# summary_all.md を生成する。
#
# 使い方 (setsid -f でdetach起動する想定、CLAUDE.md プロセス管理ルール準拠):
#   wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#     setsid -f bash scripts/_run_generalization_audit_2026-07-25.sh > logs/generalization_audit_2026-07-25.log 2>&1 < /dev/null"
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
mkdir -p logs data/verify/generalization_audit_2026-07-25

echo "[gen-audit] 開始 $(date '+%Y-%m-%d %H:%M:%S')"
bash scripts/_run_safe.sh scripts/_jobs_generalization_audit_2026-07-25.txt 3 60 3
echo "[gen-audit] 全16ジョブ完了 $(date '+%Y-%m-%d %H:%M:%S')"

echo "[gen-audit] summary_all.md 集計開始"
PYTHONPATH=. ./venv/bin/python scripts/_aggregate_generalization_audit_2026-07-25.py
echo "[gen-audit] DONE $(date '+%Y-%m-%d %H:%M:%S')"
