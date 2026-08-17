#!/usr/bin/env bash
# 一般分布ラベルセット用の構成F収集ジョブ (14本) を逐次実行する。
# MSYSパイプ・特殊文字の罠 (feedback_msys_pipe_escape.md) を避けるため
# スクリプトファイル化してある。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
mkdir -p logs
while IFS= read -r line; do
  echo "[job] ${line}"
  eval "${line}"
done < scripts/_jobs_general_yardstick_F_2026-08-17.txt
echo "ALL_JOBS_DONE"
