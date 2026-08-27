#!/bin/bash
# Gate 4 "pre-gate measurement" 条件1〜4 を逐次投入するラッパー (2026-08-25)。
#
# 【表記上の注意】本スクリプトが生成するのは Gate 4 の正式検収データではなく
# "pre-gate measurement" (Codex 許可 2026-08-25、docs/agent_coordination/
# CLAUDE_TO_CODEX.md 第25報【追記2】)。合否判定・本番ON・production 登録は
# 一切行わない。
#
# _gate4_pm100_8seg_2026-08-25.sh を条件1→3→2→4の順に呼ぶだけ
# (条件間で並列にしない=一度に触る変数を増やさない/負荷を上げない)。
# 各条件内部の区間並列度は引数 PARALLEL (既定1=逐次、上限3) をそのまま渡す。
# 条件5 は対象外 (_gate4_pm100_8seg_2026-08-25.sh 内部で exit 1 するため
# ここでは呼ばない)。
#
# 【2026-08-25 順序変更、コーディネータ指示】既定の 1→2→3→4 ではなく
# 1→3→2→4 にする。理由: user指摘の核心「規模の比較」(条件3)の評価材料を
# 条件1(基準)と揃えて最優先で確保するため。途中で止まっても価値が残る順序。
#
# 使い方 (実行判断は呼び出し側=親エージェント/userに委ねる。コマンド例のみ):
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -c \
#     "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#      setsid -f bash scripts/_run_gate4_pm100_all_2026-08-25.sh 1 \
#      > logs/gate4_pregate_pm100_2026-08-25/all.log 2>&1 < /dev/null"
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

echo "=== [pre-gate measurement] Gate4 PM100 条件1-4 一括投入 (順序1→3→2→4) start $(date +%F_%T) (並列度=$PARALLEL) ==="
for C in 1 3 2 4; do
  echo "--- 条件$C 開始 $(date +%F_%T) ---"
  bash scripts/_gate4_pm100_8seg_2026-08-25.sh "$C" 1 8 "$PARALLEL"
  RC=$?
  echo "--- 条件$C 終了 rc=$RC $(date +%F_%T) ---"
  if [ "$RC" -ne 0 ]; then
    echo "=== 中止: 条件$C が失敗した (rc=$RC) ==="
    exit "$RC"
  fi
done
echo "=== [pre-gate measurement] Gate4 PM100 条件1-4 全完了 $(date +%F_%T) ==="
echo "ALL_DONE"
