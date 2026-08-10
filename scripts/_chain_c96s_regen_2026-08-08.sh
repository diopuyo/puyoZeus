#!/usr/bin/env bash
# 第二波(49本) ALL_DONE 監視 -> c96 切り出し 3 本 (c96s1/s2/s3) の regen 起動。
# c96 は 3 カード連結の 5.5 時間動画で、 scripts/_cut_c96_series_2026-08-08.sh
# により S 級リーグ 3 シリーズへ分割済み ($HOME/frames/video_c96s{1,2,3}.mp4)。
# 動画は既に WSL 側にあるためコピーのプリステップは不要。
#
# 使い方 (WSL detach、長時間放置前提):
#   wsl -d Ubuntu -- bash -c "cd <PROJ> && \
#     setsid -f bash scripts/_chain_c96s_regen_2026-08-08.sh < /dev/null"
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1
export PYTHONPATH=.
# 1 プロセス = 1 コア厳守 (matchTemplate のスレッド競合防止)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1

WAVE2_LOG="logs/phase_l_regen_wave2_2026-08-07.log"
JOBS="scripts/_jobs_phase_l_regen_c96s_2026-08-08.txt"
LOG="logs/phase_l_regen_c96s_2026-08-08.log"
POLL_INTERVAL_SEC=300        # 5 分間隔
MAX_WAIT_SEC=$((36 * 3600))  # 最大 36 時間 (無限ループ防止)
mkdir -p logs

echo "[chain_c96s] 監視開始 $(date)" >> "${LOG}"
elapsed=0
detected=0
while [ "${elapsed}" -lt "${MAX_WAIT_SEC}" ]; do
  if [ -f "${WAVE2_LOG}" ] && grep -q "ALL_DONE" "${WAVE2_LOG}"; then
    detected=1
    echo "[chain_c96s] 第二波 ALL_DONE 検知 $(date) elapsed=${elapsed}s" >> "${LOG}"
    break
  fi
  sleep "${POLL_INTERVAL_SEC}"
  elapsed=$((elapsed + POLL_INTERVAL_SEC))
done
if [ "${detected}" -ne 1 ]; then
  echo "[chain_c96s] TIMEOUT 未起動のまま終了 $(date)" >> "${LOG}"
  exit 1
fi

# 3 本を並列実行 (1 本 ≒ 1 時間動画 × 1 コア)
n=0
while IFS= read -r job; do
  [ -z "${job}" ] && continue
  echo "[chain_c96s] launch: ${job}" >> "${LOG}"
  eval "${job}" >> "${LOG}" 2>&1 &
  n=$((n + 1))
done < "${JOBS}"
wait
echo "[chain_c96s] ALL_DONE launched=${n} $(date)" >> "${LOG}"
ls -la data/indicators_v2/boards_lean_phase_l_2026-08-07/c96s*.npz >> "${LOG}" 2>&1
