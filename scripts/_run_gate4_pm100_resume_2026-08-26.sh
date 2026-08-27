#!/bin/bash
# Gate 4 "pre-gate measurement" の再開ラッパー (2026-08-26)。
#
# 経緯: 2026-08-25 23:40 に投入した _run_gate4_pm100_all_2026-08-25.sh が
# 条件1 (8/8 rc=0) → 条件3 (8/8 rc=0) → 条件2 (seg01-03 rc=0) まで進んだ時点で、
# WSL の再起動により中断した (2026-08-26 05:23 に `uptime` が up 0 min を示し、
# 実プロセス0を確認)。
#
# 本スクリプトは既存成果物を一切上書きせず、未取得分だけを取り直す。
#   - 条件2 (cond2_hysteresis_only): seg04〜08 のみ (seg01-03 の npz は温存)
#   - 条件4 (cond4_a_plus_b): seg01〜08 (未着手)
#
# 条件1・条件3 は再実行しない (完走済み、rc=0 8/8)。
# 条件5 は Gate 3R-6 PASS まで実行禁止 (呼ばない)。
#
# 使い方:
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -c \
#     "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#      setsid -f bash scripts/_run_gate4_pm100_resume_2026-08-26.sh 3 \
#      > logs/gate4_pregate_pm100_2026-08-25/resume_2026-08-26.log 2>&1 < /dev/null"
#   (引数1個: 区間内並列度。既定1、上限3)
#
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

PARALLEL="${1:-1}"
if [ "$PARALLEL" -gt 3 ] 2>/dev/null; then
  echo "=== 中止: 並列度 $PARALLEL は上限3を超える (Codex指定) ==="
  exit 1
fi

LOGBASE=logs/gate4_pregate_pm100_2026-08-25
mkdir -p "$LOGBASE"

echo "=== [pre-gate measurement] 再開 start $(date +%F_%T) (並列度=$PARALLEL) ==="
echo "--- 条件2 seg04-08 開始 $(date +%F_%T) ---"
bash scripts/_gate4_pm100_8seg_2026-08-25.sh 2 4 8 "$PARALLEL"
RC=$?
echo "--- 条件2 seg04-08 終了 rc=$RC $(date +%F_%T) ---"
if [ "$RC" -ne 0 ]; then
  echo "=== 中止: 条件2 が失敗した (rc=$RC) ==="
  exit "$RC"
fi

echo "--- 条件4 seg01-08 開始 $(date +%F_%T) ---"
bash scripts/_gate4_pm100_8seg_2026-08-25.sh 4 1 8 "$PARALLEL"
RC=$?
echo "--- 条件4 seg01-08 終了 rc=$RC $(date +%F_%T) ---"
if [ "$RC" -ne 0 ]; then
  echo "=== 中止: 条件4 が失敗した (rc=$RC) ==="
  exit "$RC"
fi

echo "=== [pre-gate measurement] 条件1-4 全完了 $(date +%F_%T) ==="
echo "RESUME_ALL_DONE"
