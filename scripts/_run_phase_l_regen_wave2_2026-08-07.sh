#!/bin/bash
# Phase L 第二波(DL拡大分)regen ランナー (2026-08-07)
# 第一波 (scripts/_run_phase_l_regen_2026-08-07.sh) と同構成。14コア飽和の
# ため第一波完了後にのみ起動すること (scripts/_chain_wave2_2026-08-07.sh が
# 自動判定する)。対象は scripts/_jobs_phase_l_regen_wave2_2026-08-07.txt
# (c96-c144のDL成功分、on_disk基準で動的生成、45本前後想定)。
#
# 使い方 (WSL detach で長時間放置前提):
#   wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#     setsid -f bash scripts/_run_phase_l_regen_wave2_2026-08-07.sh [MAXPAR=14] < /dev/null"
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
# 1プロセス=1コアを厳守 (matchTemplate 系スレッド競合防止、
# memory `project_collect_indicators_v2_perf_2026-07-20` 教訓)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

JOBS="scripts/_jobs_phase_l_regen_wave2_2026-08-07.txt"
MAXPAR="${1:-14}"
OUT_DIR="data/indicators_v2/boards_lean_phase_l_2026-08-07"
LOG="logs/phase_l_regen_wave2_2026-08-07.log"
mkdir -p "${OUT_DIR}" logs

# --- WSL側動画同期プリステップ (第一波と同じロジック) ---
# ジョブは $HOME/frames/<name>.mp4 を読むが、$HOME/frames には過去ジョブで
# コピーされた分しか無く全動画分ではない。ジョブ一覧が実際に参照する動画の
# うち $HOME/frames に無い、またはサイズ0のものだけを Windows側 data/frames/
# (WSLからは /mnt/c/... 経由) から cp する。既存分は re-copy せず skip する。
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
echo "[phase_l_regen_wave2] 開始 $(date) ジョブ数=${EXPECTED} MAXPAR=${MAXPAR}" >> "${LOG}"
n=0
while IFS= read -r cmd; do
  [ -z "$cmd" ] && continue
  case "$cmd" in \#*) continue ;; esac
  # 空きスロットが出るまで待つ (=前ジョブ完了待ち)
  while [ "$(jobs -r | wc -l)" -ge "$MAXPAR" ]; do sleep 5; done
  n=$((n+1))
  echo "[phase_l_regen_wave2 $(date '+%H:%M:%S')] (${n}/${EXPECTED}) start: ${cmd}" >> "${LOG}"
  nice -n 19 bash -c "${cmd}" >> "${LOG}" 2>&1 &
done < "${JOBS}"
wait

ACTUAL=$(find "${OUT_DIR}" -maxdepth 1 -name '*.npz' | wc -l)
echo "[phase_l_regen_wave2] ALL_DONE $(date) jobs_run=${n} expected_npz=${EXPECTED} actual_npz(全体)=${ACTUAL}" >> "${LOG}"
