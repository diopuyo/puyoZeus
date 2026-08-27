#!/bin/bash
# zenchi chain_id 対応表検収の実行ラッパ (2026-08-25)。
# MSYS パイプ・エスケープ問題の回避のためスクリプトファイル化
# (memory feedback_msys_pipe_escape)。
# 使い方: wsl bash scripts/_run_zenchi_chainid_2026-08-25.sh <t0> <t1> <prefix>
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
T0="$1"
T1="$2"
PREFIX="$3"
OUT_DIR=data/verify/gate3_zenchi_chainid_2026-08-25
mkdir -p "$OUT_DIR"
START=$(date +%s)
PYTHONPATH=. nice -n 19 ./venv/bin/python scripts/_gate3_rate_trace_2026-08-25.py \
  --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 \
  --t0 "$T0" --t1 "$T1" \
  --out-dir "$OUT_DIR" --out-prefix "$PREFIX" \
  > "logs/_gate3_zenchi_${PREFIX}_2026-08-25.log" 2>&1
RC=$?
END=$(date +%s)
echo "exit=$RC elapsed=$((END - START))s"
