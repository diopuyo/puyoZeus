#!/bin/bash
# 分割レンダ+ウォームアップの一致検証 (2026-08-21)。
# scripts/visualize_advantage_overlay.py は変更しない。CLI 呼び出しのみ。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p data/verify/zenchi_warmup_2026-08-21
LOG=logs/_diag_warmup_consistency_2026-08-21.log
{ date; cat /proc/loadavg; } > "$LOG"

run_one() {
  name="$1"; start="$2"; end="$3"; warm="$4"
  echo "[run] name=$name start=$start end=$end warmup=$warm" >> "$LOG"
  t0=$(date +%s)
  PYTHONPATH=. ./venv/bin/python -u -m scripts.visualize_advantage_overlay \
    --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 \
    --start-sec "$start" --end-sec "$end" --warmup-sec "$warm" \
    --no-force-in-match --model-dir data/verify/retrain_model62_2026-08-21 \
    --no-render --out "data/verify/zenchi_warmup_2026-08-21/${name}.mp4" \
    --dump-timeline "data/verify/zenchi_warmup_2026-08-21/${name}.npz" \
    >> "$LOG" 2>&1
  t1=$(date +%s)
  echo "[run done] name=$name elapsed=$((t1-t0))s" >> "$LOG"
}

run_one ref 3300 3360 0
run_one w0 3326 3360 0
run_one w5 3326 3360 5
run_one w15 3326 3360 15
run_one w26 3326 3360 26

{ date; cat /proc/loadavg; } >> "$LOG"
echo ALL_DONE >> "$LOG"
