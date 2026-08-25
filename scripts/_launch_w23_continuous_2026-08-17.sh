#!/bin/bash
# 持続誤認70件・連続処理検証の6グループを個別に detach 起動する (2026-08-17)。
# MSYSパイプ/変数escape事故 (feedback_msys_pipe_escape) を避けるためスクリプトファイル化。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
mkdir -p logs/diag_w23_continuous_2026-08-17

for g in c17_chunk0 c13_chunk0 c13_chunk1 c21_chunk1 c21_chunk2 c22_chunk1; do
  setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -u -m scripts._diag_w23_continuous_verify_2026-08-17 --group ${g} > logs/diag_w23_continuous_2026-08-17/${g}.log 2>&1 < /dev/null"
  echo "launched: ${g}"
done
