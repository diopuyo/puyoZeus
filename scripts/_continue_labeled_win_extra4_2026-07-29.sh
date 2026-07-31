#!/bin/bash
# 打ち合い時間モデル用4動画(c5,c30,c31,c83) labeled_win収集の自動チェイン (2026-07-29)。
# ジョブ生成 (_gen_jobs_labeled_win_extra4_2026-07-29.sh) 完了後に呼ばれる前提。
# 14ジョブ (c5:4窓, c30:4窓, c31:3窓, c83:3窓) を高並列 (MAXPAR=14, cv2 1スレッド)
# で消化し、完了検知 (全出力揃い) 後に行数サニティチェックまで自動実行する
# (fail-silent 厳禁: 出力欠落・行数異常があれば ERROR を明示してラベル結合等の
#  後続処理には進まない)。
#
# 並列数14はc20/m20/m30実績の最適値
# (memory: project_collect_indicators_v2_perf_2026-07-20)。
#
# 注意: 本タスクの目的は着弾遅延の物理計測 (打ち合い時間モデル) であり、
# 勝敗ラベル学習ではないため label_win_from_winners への結合は行わない
# (winners_panel_diff_gated_2026-07-26 に4動画分のJSONは存在することを確認済みだが、
#  今回の用途では不要なためスコープ外とした)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.

OUT_DIR="data/verify/labeled_win_extra4_2026-07-29"
LOG="logs/orchestrate_labeled_win_extra4_2026-07-29.log"
JOBS="scripts/_jobs_labeled_win_extra4_2026-07-29.txt"
MIN_ROWS=100  # 出力csvが空/極端に少ない場合の異常検知しきい値 (フレーム全数収集前提)
mkdir -p "${OUT_DIR}/study" logs

EXPECTED=$(wc -l < "${JOBS}")
echo "[extra4] 開始 $(date) MAXPAR=14 THREADS=1 EXPECTED=${EXPECTED}" >> "${LOG}"

# --- ${EXPECTED}ジョブを高並列(MAXPAR=14, THREADS=1, COOLDOWN=5)で消化 ---
bash scripts/_run_safe.sh "${JOBS}" 14 5 1 >> "${LOG}" 2>&1
echo "[extra4] 収集ジョブ完了 $(date)" >> "${LOG}"

# --- 完了検知: 全出力(csv)が揃うまで確認 (通常は上記で既に揃っているはず) ---
for i in $(seq 1 60); do
  actual=$(ls "${OUT_DIR}/study/"*.csv 2>/dev/null | wc -l)
  if [ "${actual}" -ge "${EXPECTED}" ]; then
    echo "[extra4] 全${EXPECTED}出力確認 $(date)" >> "${LOG}"
    break
  fi
  echo "[extra4 $(date '+%H:%M:%S')] 出力確認 ${actual}/${EXPECTED}、待機継続" >> "${LOG}"
  sleep 60
done

# --- fail-silent防止: 出力csvの行数サニティチェック ---
fail=0
actual=$(ls "${OUT_DIR}/study/"*.csv 2>/dev/null | wc -l)
if [ "${actual}" -lt "${EXPECTED}" ]; then
  echo "[ERROR] 出力数不足: ${actual}/${EXPECTED} (60分待機後も未完走)" >> "${LOG}"
  fail=1
fi
for f in "${OUT_DIR}/study/"*.csv; do
  [ -f "$f" ] || continue
  n=$(wc -l < "$f")
  if [ "$n" -lt "$MIN_ROWS" ]; then
    echo "[ERROR] $f の行数が異常に少ない (${n}行 < ${MIN_ROWS}行、動画読込失敗または未反映の疑い)" >> "${LOG}"
    fail=1
  fi
done

if [ "${fail}" -eq 1 ]; then
  echo "[extra4] ERROR DONE $(date) (異常あり、要調査)" >> "${LOG}"
  exit 1
fi

echo "[extra4] 全出力サニティチェックOK $(date)" >> "${LOG}"
echo "[extra4] ALL DONE $(date)" >> "${LOG}"
