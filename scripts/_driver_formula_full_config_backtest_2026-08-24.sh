#!/bin/bash
# Q-03 是正: 「完全構成」バックテスト (2026-08-24)。
#
# 背景: 既存の data/verify/formula_read_backtest_2026-08-24/ は、複合フラグ
# --enable-formula-freeze-fix だけを渡していたため、3 要素目
# enable_slide_exit_no_min_display が実質 no-op だった
# (src/recognition_pipeline.py:5171-5203 で親ガード
#  enable_slide_exit_min_display_guard=False なら子は一度も参照されない)。
#
# 本ドライバは親ガードを明示 ON にした「完全構成」を、
# **別出力先** に測定する。既存成果物は一切上書きしない。
#
# 条件:
#   F1  親ガードのみ       --enable-slide-exit-min-display-guard
#   F2  完全構成           --enable-slide-exit-min-display-guard --enable-formula-freeze-fix
#
# 比較の基準 (OFF / 式読取2要素のみ) は既存ディレクトリの
#   off_baseline_result.json / on_result.json をそのまま使う (再測定しない)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
PY="./venv/bin/python"
OUT=data/verify/formula_read_backtest_full_2026-08-24
LOGD=logs/formula_read_backtest_full_2026-08-24
ARGS="--videos v29,v40,v51,v57,v70,v89,v95,v97 --holdout v89,v97 --workers 4 --game-event-chain-exit"

mkdir -p "$OUT" "$LOGD"

# F1: 親ガードのみ (no-min は OFF = 従来の 0.8 秒最小表示ガードが効く)
PYTHONPATH=. nice -n 5 $PY scripts/measure_stable_cell_acc.py $ARGS \
  --enable-slide-exit-min-display-guard \
  --output "$OUT/f1_slide_guard_only_result.json" \
  > "$LOGD/f1_slide_guard_only.log" 2>&1
echo "done f1" >> "$LOGD/progress.log"

# F2: 完全構成 (親ガード ON + 式読取2要素 + no-min)
PYTHONPATH=. nice -n 5 $PY scripts/measure_stable_cell_acc.py $ARGS \
  --enable-slide-exit-min-display-guard --enable-formula-freeze-fix \
  --output "$OUT/f2_full_config_result.json" \
  > "$LOGD/f2_full_config.log" 2>&1
echo "done f2" >> "$LOGD/progress.log"

md5sum "$OUT"/*.json > "$OUT/md5sums.txt"
echo FULL_CONFIG_BACKTEST_ALL_DONE >> "$OUT/md5sums.txt"
echo "ALL_DONE" >> "$LOGD/progress.log"
