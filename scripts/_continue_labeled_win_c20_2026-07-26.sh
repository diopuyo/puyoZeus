#!/bin/bash
# #43 c系20本 labeled_win 収集: 並列引き上げ継続ランナー (2026-07-27)。
# 旧オーケストレータ(_orchestrate_labeled_win_c20_2026-07-26.sh + 旧 _run_safe.sh MAXPAR=3)を
# ジョブ境界で安全停止した後を引き継ぐ。既に完了/実行中の14ジョブは対象外、
# 残り46ジョブ(scripts/_jobs_labeled_win_c20_2026-07-26_remaining.txt)のみを
# 高並列(MAXPAR=14, cv2 1スレッド)で消化し、完了後にラベル結合まで自動実行する。
#
# 引き上げ根拠: PCクーラー導入によりCPU温度対策は不要と判断(2026-07-27 user指示)。
# 並列数14は過去実績の最適値 (memory: project_collect_indicators_v2_perf_2026-07-20,
# cv2.setNumThreads(1)ラッパー×14並列)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.

OUT_DIR="data/verify/labeled_win_c20_2026-07-26"
LOG="logs/orchestrate_labeled_win_c20_2026-07-26.log"
REMAINING_JOBS="scripts/_jobs_labeled_win_c20_2026-07-26_remaining.txt"

echo "[continue] 並列引き上げ継続ランナー開始 $(date) MAXPAR=14 THREADS=1" >> "${LOG}"

# --- 残り46ジョブを高並列(MAXPAR=14, THREADS=1, COOLDOWN=5)で消化 ---
bash scripts/_run_safe.sh "${REMAINING_JOBS}" 14 5 1 >> "${LOG}" 2>&1
echo "[continue] 残ジョブ完了 $(date)" >> "${LOG}"

# --- 旧runner側で実行中だった3ジョブ(c13_gap/c14/c14_gap)の完了待ち ---
# (通常は上記46ジョブより先に終わっているはずだが、念のため全60出力を確認してから結合)
EXPECTED=60
for i in $(seq 1 120); do
  actual=$(ls "${OUT_DIR}/study/"*.csv 2>/dev/null | wc -l)
  if [ "${actual}" -ge "${EXPECTED}" ]; then
    echo "[continue] 全${EXPECTED}出力確認 $(date)" >> "${LOG}"
    break
  fi
  echo "[continue $(date '+%H:%M:%S')] 出力確認 ${actual}/${EXPECTED}、待機継続" >> "${LOG}"
  sleep 60
done

# --- ラベル結合 (旧オーケストレータStep4を継承) ---
./venv/bin/python -m scripts.label_win_from_winners \
  --study "${OUT_DIR}/study" \
  --winners-dir data/verify/winners_panel_diff_gated_2026-07-26 \
  --out "${OUT_DIR}/labeled_win_c20.csv" >> "${LOG}" 2>&1

echo "[continue] ALL DONE $(date)" >> "${LOG}"
