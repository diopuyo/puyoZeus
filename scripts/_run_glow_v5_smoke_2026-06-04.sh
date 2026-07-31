#!/bin/bash
# v5 スモーク: v89_match01 の board_log jsonl のみ生成し、v4 と比較する。
# v5 の変更: apply_glow_guard の frozen非有色でもconsensus復元 (frozen=O/空/UNKNOWNでも CNN==HSV=色 → 復元)
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
VDIR="data/match_clips"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"

echo "[start] glow v5 smoke $(date)"

# v5 の board_log jsonl 生成 (v4 と同じフラグ構成、v5 コードで実行)
PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
  --video "${VDIR}/v89/v89_match01.mp4" \
  --output "${VIZDIR}/v89_match01_glowV5_2026-06-04.mp4" \
  --ojama-warning-glow-guard \
  --dump-board-log-detailed "${VIZDIR}/v89_match01_glowV5_2026-06-04.jsonl" \
  > "${LOGDIR}/viz_v89_glowV5.log" 2>&1
echo "[viz] v5 board_log 生成完了 $(date)"

# v4 vs v5 スモーク比較
PYTHONPATH=. venv/bin/python -m scripts.smoke_glow_v5 \
  --v4-log "${VIZDIR}/v89_match01_glowV4_2026-06-04.jsonl" \
  --v5-log "${VIZDIR}/v89_match01_glowV5_2026-06-04.jsonl" \
  --t-start 66 --t-end 72 \
  > "${LOGDIR}/smoke_glow_v5.log" 2>&1
echo "[smoke] v4 vs v5 比較完了 $(date)"
echo "[done] smoke_glow_v5.log を確認してください"
