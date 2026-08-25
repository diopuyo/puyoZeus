#!/bin/bash
# STABLE凍結デッドロック根治: 物差し3本の逐次ドライバ (2026-08-24 コーダ)。
# 前提: scripts/_driver_formula_fix_verify_2026-08-24.sh が ALL_DONE を出すまで待つ。
#  A. 基準 (変更前コード = worktree 再構築版)、フラグOFF
#  B. 最終コード、フラグOFF   → A と md5 一致で bit-identical 証明
#  C. 最終コード、フラグON    → 認識精度 99.5% ライン維持の確認
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
PY="./venv/bin/python"
WT="/mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/85af3971-d05b-42eb-8a0a-fce931916160/scratchpad/wt_baseline_ff"
OUT=data/verify/formula_read_backtest_2026-08-24
ARGS="--videos v29,v40,v51,v57,v70,v89,v95,v97 --holdout v89,v97 --workers 4 --game-event-chain-exit"

# 第1ドライバの完了待ち
until grep -q ALL_DONE logs/_probe_formula_false_event_2026-08-24/driver_progress.log 2>/dev/null; do
  sleep 60
done

# A. 基準 (worktree の src を PYTHONPATH 先頭に)
PYTHONPATH="$WT" nice -n 5 $PY "$WT/scripts/measure_stable_cell_acc.py" $ARGS \
  --output $OUT/off_baseline_result.json \
  > logs/formula_read_backtest_2026-08-24_off_baseline.log 2>&1

# B. 最終コード OFF
PYTHONPATH=. nice -n 5 $PY scripts/measure_stable_cell_acc.py $ARGS \
  --output $OUT/off_final_result.json \
  > logs/formula_read_backtest_2026-08-24_off_final.log 2>&1

# C. 最終コード ON
PYTHONPATH=. nice -n 5 $PY scripts/measure_stable_cell_acc.py $ARGS \
  --enable-formula-freeze-fix \
  --output $OUT/on_result.json \
  > logs/formula_read_backtest_2026-08-24_on.log 2>&1

md5sum $OUT/*.json > $OUT/md5sums.txt
echo BACKTEST_ALL_DONE >> $OUT/md5sums.txt
