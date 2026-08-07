#!/bin/bash
# Phase L 全動画regen ランナー (2026-08-07)
# 新標準構成 (第4機構修正A' 含む) で対象動画 (97本、tier白リスト
# S級/マスター/チャレンジャー/A級のみ、その他大会系は2026-08-07 user指示で
# 既定除外) を一括 board 抽出する。
# ジョブ生成元: scripts/_gen_jobs_phase_l_regen_2026-08-07.py
#   (対象選定ロジック = scripts/build_video_tier_index.py, ID重複解消済 +
#    PHASE_L_TIER_WHITELIST フィルタ)
#
# 使い方 (WSL detach で長時間放置前提):
#   wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#     setsid -f bash scripts/_run_phase_l_regen_2026-08-07.sh [MAXPAR=14] < /dev/null"
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
# 1プロセス=1コアを厳守 (matchTemplate 系スレッド競合防止、
# memory `project_collect_indicators_v2_perf_2026-07-20` 教訓)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_phase_l_regen_2026-08-07.txt"
MAXPAR="${1:-14}"
OUT_DIR="data/indicators_v2/boards_lean_phase_l_2026-08-07"
LOG="logs/phase_l_regen_2026-08-07.log"
mkdir -p "${OUT_DIR}" logs

# --- WSL側動画同期プリステップ (2026-08-07 追加) ---
# ジョブは $HOME/frames/<name>.mp4 を読むが、$HOME/frames には過去ジョブで
# コピーされた分しか無く全動画分ではない (2026-08-07時点 80/97本確認)。
# ジョブ一覧が実際に参照する動画のうち $HOME/frames に無い、または
# サイズ0のものだけを Windows側 data/frames/ (WSLからは /mnt/c/... 経由)
# から cp する。既存分は re-copy せず skip する (ネットワーク越しI/O節約)。
WIN_FRAMES_DIR="${PROJ_DIR}/data/frames"
mkdir -p "$HOME/frames"
echo "[sync] 開始 $(date)" >> "${LOG}"
n_copied=0
n_skipped=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  match=$(printf '%s' "$cmd" | grep -oE 'frames/[^ ]+\.mp4' | head -1)
  [ -z "$match" ] && continue
  vname="${match#frames/}"
  dst="$HOME/frames/${vname}"
  src="${WIN_FRAMES_DIR}/${vname}"
  dst_size=0
  [ -f "$dst" ] && dst_size=$(stat -c%s "$dst" 2>/dev/null || echo 0)
  if [ "$dst_size" -gt 0 ]; then
    n_skipped=$((n_skipped+1))
    continue
  fi
  if [ ! -f "$src" ]; then
    echo "[sync][ERROR] source missing: ${src}" >> "${LOG}"
    continue
  fi
  echo "[sync] copy ${vname} $(date '+%H:%M:%S')" >> "${LOG}"
  cp -f "$src" "$dst"
  n_copied=$((n_copied+1))
done < "${JOBS}"
echo "[sync] 完了 $(date) copied=${n_copied} skipped(既存)=${n_skipped}" >> "${LOG}"

EXPECTED=$(grep -c . "${JOBS}")
echo "[phase_l_regen] 開始 $(date) ジョブ数=${EXPECTED} MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  # 空きスロットが出るまで待つ (=前ジョブ完了待ち)
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[phase_l_regen $(date '+%H:%M:%S')] (${n}/${EXPECTED}) start: ${cmd}" >> "${LOG}"
  nice -n 19 bash -c "${cmd}" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait

ACTUAL=$(find "${OUT_DIR}" -maxdepth 1 -name '*.npz' | wc -l)
echo "[phase_l_regen] ALL_DONE $(date) jobs_run=${n} expected_npz=${EXPECTED} actual_npz=${ACTUAL}" >> "${LOG}"
