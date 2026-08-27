#!/bin/bash
# Gate 4 "pre-gate measurement" 条件1〜4 の集計 (2026-08-25)。
#
# 【表記上の注意】本スクリプトが出す数値は Gate 4 の正式検収データではなく
# "pre-gate measurement" (Codex 許可 2026-08-25、docs/agent_coordination/
# CLAUDE_TO_CODEX.md 第25報【追記2】)。合否判定はしない(数値を出すだけ)。
#
# 既存の集計器 scripts/_analyze_pm100_pair_2026-08-24.py (無改修で再利用、
# 母数付きで①張り付き ②符号逆 ③反転試合 ④急変 受け入れ条件4=決着方向誤り・
# 弱化試合 を出す) を条件1(基準)と条件2/3/4の各ペアに対して呼ぶだけ。
# stdlib + numpy のみに依存し src/scripts の他モジュールを import しないため
# (2026-08-25 確認済み)、固定 snapshot 側のコピーをそのまま使う
# (作業ツリーの変化に一切依存しない)。
#
# 【重要】合格条件9項目のうち、この集計器がカバーするのは実測で5項目
# (①seg01 game2 個別確認 / ⑤±100張り付き / ⑥生モデル逆符号張り付き /
#  ⑦決着方向誤り / ⑧真の致死局面を弱めた試合)。
# 残り4項目 (②交換保存則 ③二重計上 ④急変の残り全件が新物理イベントか の
# 目視確認 ⑨認識精度+全pytest) はこのスクリプトでは測れない。詳細は
# コーダの報告 (親エージェントへの回答) を参照。
#
# 使い方 (実行判断は呼び出し側=親エージェント/userに委ねる):
#   bash scripts/_gate4_pm100_aggregate_2026-08-25.sh
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

SNAP=data/verify/gate4_pregate_pm100_2026-08-25/_snapshot_20260825_2300jst
if [ ! -f "$SNAP/scripts/_analyze_pm100_pair_2026-08-24.py" ]; then
  echo "=== 中止: snapshot が見つからない ($SNAP) ==="
  exit 1
fi

OUTBASE=data/verify/gate4_pregate_pm100_2026-08-25
LOGDIR=logs/gate4_pregate_pm100_2026-08-25
mkdir -p "$LOGDIR"

BASE="$OUTBASE/cond1_off_baseline"

echo "=== [pre-gate measurement] 集計開始 $(date +%F_%T) ==="
for C in 2 3 4; do
  case "$C" in
    2) NAME="cond2_hysteresis_only" ;;
    3) NAME="cond3_scale_compare_only" ;;
    4) NAME="cond4_a_plus_b" ;;
  esac
  ON="$OUTBASE/$NAME"
  echo "=== [pre-gate measurement] 条件1(基準) vs 条件$C ($NAME) ==="
  if [ ! -d "$BASE" ] || [ ! -d "$ON" ]; then
    echo "  スキップ: dump ディレクトリが無い ($BASE または $ON)"
    continue
  fi
  nice -n 19 ./venv/bin/python "$SNAP/scripts/_analyze_pm100_pair_2026-08-24.py" \
    "$BASE" "$ON" | tee "$LOGDIR/aggregate_cond1_vs_cond${C}.log"
  echo
done
echo "=== [pre-gate measurement] 集計完了 $(date +%F_%T) (合否判定はしない) ==="
