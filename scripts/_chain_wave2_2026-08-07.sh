#!/bin/bash
# 第一波(97本) ALL_DONE 監視 -> 台帳/wave2ジョブ再生成 -> 第二波起動 チェーン
# (2026-08-07)。14コア飽和のため第一波完了を待って第二波(DL拡大分・c96-c144)
# を起動する必要がある。5分間隔でポーリングし、最大48時間でタイムアウト
# (起動せず記録して終了、無限ループ防止)。
#
# 再生成時に再試行DL (c104/c113/c123/c141等) の成否を反映するため、
# 台帳再生成(build_video_tier_index.py) と wave2ジョブ再生成
# (_gen_jobs_phase_l_regen_2026-08-07.py --exclude-jobs <第一波> --out <wave2>)
# をこのタイミングで実行する。第一波ジョブファイルは --exclude-jobs でのみ
# 参照し、書き込みは一切行わない (絶対不変)。
#
# 使い方 (WSL detach で長時間放置前提):
#   wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && \
#     setsid -f bash scripts/_chain_wave2_2026-08-07.sh < /dev/null"
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}" || exit 1

WAVE1_LOG="logs/phase_l_regen_2026-08-07.log"
CHAIN_LOG="logs/chain_wave2_2026-08-07.log"
POLL_INTERVAL_SEC=300       # 5分間隔でポーリング
MAX_WAIT_SEC=$((48 * 3600)) # 最大48時間待機 (無限ループ防止)
WAVE1_JOBS="scripts/_jobs_phase_l_regen_2026-08-07.txt"
WAVE2_JOBS="scripts/_jobs_phase_l_regen_wave2_2026-08-07.txt"
WAVE2_RUNNER="scripts/_run_phase_l_regen_wave2_2026-08-07.sh"

mkdir -p logs
export PYTHONPATH=.
echo "[chain_wave2] 監視開始 $(date) poll=${POLL_INTERVAL_SEC}s timeout=${MAX_WAIT_SEC}s" >> "${CHAIN_LOG}"

elapsed=0
detected=0
while [ "${elapsed}" -lt "${MAX_WAIT_SEC}" ]; do
  if [ -f "${WAVE1_LOG}" ] && grep -q "ALL_DONE" "${WAVE1_LOG}"; then
    detected=1
    echo "[chain_wave2] 第一波 ALL_DONE 検知 $(date) elapsed=${elapsed}s" >> "${CHAIN_LOG}"
    break
  fi
  sleep "${POLL_INTERVAL_SEC}"
  elapsed=$((elapsed + POLL_INTERVAL_SEC))
done

if [ "${detected}" -eq 0 ]; then
  echo "[chain_wave2] TIMEOUT $(date) ${MAX_WAIT_SEC}秒待機してもALL_DONE未検知。第二波は起動せず終了。" >> "${CHAIN_LOG}"
  exit 1
fi

# (a) 再試行DLの成否を反映してジョブ生成をやり直す
echo "[chain_wave2] 台帳再生成開始 $(date)" >> "${CHAIN_LOG}"
./venv/bin/python -u scripts/build_video_tier_index.py >> "${CHAIN_LOG}" 2>&1
./venv/bin/python -u scripts/_gen_jobs_phase_l_regen_2026-08-07.py \
  --exclude-jobs "${WAVE1_JOBS}" --out "${WAVE2_JOBS}" >> "${CHAIN_LOG}" 2>&1
WAVE2_COUNT=$(grep -c . "${WAVE2_JOBS}" 2>/dev/null || echo 0)
echo "[chain_wave2] wave2ジョブ再生成完了 $(date) ${WAVE2_COUNT}本 -> ${WAVE2_JOBS}" >> "${CHAIN_LOG}"

# (b) 第二波ランナー起動
echo "[chain_wave2] 第二波起動 $(date)" >> "${CHAIN_LOG}"
bash "${WAVE2_RUNNER}" >> "${CHAIN_LOG}" 2>&1
echo "[chain_wave2] CHAIN_DONE $(date)" >> "${CHAIN_LOG}"
