#!/bin/bash
# STABLE凍結デッドロック根治 (2026-08-24): フラグ既定OFFの bit-identical 検証。
# 2026-08-22 の slide_exit_guard_backtest と同一引数で物差し測定を再実行し、
# data/verify/slide_exit_guard_backtest_2026-08-22/off_result.json
# (md5=514ce6e2c16551ba721e64a675e2edb0) と md5 一致することを確認する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
PYTHONPATH=. nice -n 5 ./venv/bin/python scripts/measure_stable_cell_acc.py \
  --videos v29,v40,v51,v57,v70,v89,v95,v97 \
  --holdout v89,v97 \
  --workers 4 \
  --game-event-chain-exit \
  --output data/verify/formula_read_backtest_2026-08-24/off_result.json \
  > logs/formula_read_backtest_2026-08-24_off.log 2>&1
