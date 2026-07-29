#!/bin/bash
# warmup A/B比較 (2026-07-29): DriftDetector 再同期暴走ガードの実効窓が
# warmup有無でずれる仮説 (_match_active_started_time が処理開始時刻に
# 引きずられる) を _diag_settle_freeze_2026-07-29.py の計装出力で検証する。
# nice -n 19、逐次実行 (並列にしない、CLAUDE.mdプロセス管理ルール準拠)。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.

LOG=logs/diag_warmup_ab_2026-07-29.log
mkdir -p logs
log() { echo "[warmup-ab] $(date) $*" | tee -a "${LOG}"; }

log "起動 (PID=$$)。video_c56 c56_g3 区間 (start=288 end=345) を warmup=0 → warmup=30 の順で逐次診断。"

log "[1/2] warmup=0 診断開始"
nice -n 19 ./venv/bin/python -u -m scripts._diag_settle_freeze_2026-07-29 \
  --video data/frames/video_c56.mp4 --start-sec 288 --end-sec 345 \
  --warmup-sec 0 --label c56_g3_warmup0 \
  > logs/_diag_c56_g3_warmup0_2026-07-29.log 2>&1
log "[1/2] warmup=0 診断完了"

log "[2/2] warmup=30 診断開始"
nice -n 19 ./venv/bin/python -u -m scripts._diag_settle_freeze_2026-07-29 \
  --video data/frames/video_c56.mp4 --start-sec 288 --end-sec 345 \
  --warmup-sec 30 --label c56_g3_warmup30 \
  > logs/_diag_c56_g3_warmup30_2026-07-29.log 2>&1
log "[2/2] warmup=30 診断完了"

log "ALL DONE"
