#!/bin/bash
# #43 段階2: マスター級20本 labeled_win 収集の自動チェイン (2026-07-28)。
# ジョブ生成(_gen_jobs_labeled_win_m20_2026-07-28.sh)完了後に呼ばれる前提。
# 60ジョブ(3窓 x 20本)を高並列(MAXPAR=14, cv2 1スレッド)で消化し、
# 完了検知(全60出力揃い)後にラベル結合まで自動実行する。
#
# 並列数14はc20実績の最適値 (memory: project_collect_indicators_v2_perf_2026-07-20,
# cv2.setNumThreads(1)ラッパー×14並列)。PCクーラー導入済みのため熱対策は不要
# (2026-07-27 user指示、c20継続ランナーで採用済みの構成を踏襲)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.

OUT_DIR="data/verify/labeled_win_m20_2026-07-28"
LOG="logs/orchestrate_labeled_win_m20_2026-07-28.log"
JOBS="scripts/_jobs_labeled_win_m20_2026-07-28.txt"
mkdir -p "${OUT_DIR}/study" logs

echo "[m20] 開始 $(date) MAXPAR=14 THREADS=1" >> "${LOG}"

# --- 60ジョブを高並列(MAXPAR=14, THREADS=1, COOLDOWN=5)で消化 ---
bash scripts/_run_safe.sh "${JOBS}" 14 5 1 >> "${LOG}" 2>&1
echo "[m20] 収集ジョブ完了 $(date)" >> "${LOG}"

# --- 完了検知: 全60出力(csv)が揃うまで確認 (通常は上記で既に揃っているはず) ---
EXPECTED=60
for i in $(seq 1 60); do
  actual=$(ls "${OUT_DIR}/study/"*.csv 2>/dev/null | wc -l)
  if [ "${actual}" -ge "${EXPECTED}" ]; then
    echo "[m20] 全${EXPECTED}出力確認 $(date)" >> "${LOG}"
    break
  fi
  echo "[m20 $(date '+%H:%M:%S')] 出力確認 ${actual}/${EXPECTED}、待機継続" >> "${LOG}"
  sleep 60
done

# --- ラベル結合 (c20継続ランナーStep4を踏襲) ---
./venv/bin/python -m scripts.label_win_from_winners \
  --study "${OUT_DIR}/study" \
  --winners-dir data/verify/winners_panel_diff_gated_2026-07-26 \
  --out "${OUT_DIR}/labeled_win_m20.csv" >> "${LOG}" 2>&1

echo "[m20] ALL DONE $(date)" >> "${LOG}"
