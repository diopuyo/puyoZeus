#!/bin/bash
# 同値確認用: 配布物の app/ をそのまま使い、WSL の開発 venv (Linux) で既存パイプラインを流す。
# 配布版 (Windows) と「同一ソース・同一資産・同一 CLI」にして、差を実行環境 (OS・torch・numpy・puyo_core) に限る。
#
#   wsl -d Ubuntu -- bash /mnt/c/.../packaging/run_equiv.sh <ラベル> <動画(WSLパス)> <開始秒> <終了秒>
#
# 出力: /mnt/d/puyo_analyzer/packaging/equiv/<ラベル>/ (display.npz / settled.npz / events.jsonl / review_data.csv ほか)
# 比較: Windows 側で  python packaging/compare_outputs.py <配布版の出力> <この出力>
set -euo pipefail
LABEL="${1:?ラベル}"
VIDEO="${2:?動画パス}"
START="${3:?開始秒}"
END="${4:?終了秒}"
APP=/mnt/d/puyo_analyzer/packaging/build/PuyoLive/app
PY=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/bin/python
OUT="/mnt/d/puyo_analyzer/packaging/equiv/${LABEL}"
rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${APP}"
# 配布版の app を import 元にする。CUDA は無効 (既定の cnn_device=cpu と同じ)。
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${APP}" CUDA_VISIBLE_DEVICES="" "${PY}" -m scripts.run_live_pipeline_20260928 \
  --video "${VIDEO}" --start-sec "${START}" --end-sec "${END}" --warmup-sec 1 \
  --output "${OUT}" --no-split-evaluation --no-async-counter --port 8792 \
  > "${OUT}.log" 2>&1
echo "done: ${OUT}"
