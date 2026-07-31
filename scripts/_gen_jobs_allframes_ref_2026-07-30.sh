#!/bin/bash
# 全フレーム基準データ収集 (2026-07-30) のジョブ生成ラッパー。
#
# 選定ロジック(skip先頭3試合 + strict限定 + 試合長[30,120]秒 + 等間隔5本選定)は
# scripts/_select_allframes_ref_games_2026-07-30.py (stdlib のみ、型ヒント付き) に
# 実装済み。本スクリプトはそれを呼び出し、結果を確認表示するだけの薄いラッパー。
#
# 出力:
#   - data/verify/allframes_ref_2026-07-30/selected_games.csv (選定レポート)
#   - scripts/_jobs_allframes_ref_2026-07-30.txt (_collect_lean_1t ジョブ定義)
#
# 使い方: bash scripts/_gen_jobs_allframes_ref_2026-07-30.sh
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1

python3 -m scripts._select_allframes_ref_games_2026-07-30
STATUS=$?

echo "[gen] ジョブ数: $(wc -l < scripts/_jobs_allframes_ref_2026-07-30.txt)"
exit "${STATUS}"
