#!/bin/bash
# 8分割点 (1/8,2/8,...,7/8地点) 付近の試合開始時刻をスポット確認する (2026-08-21)。
# --no-render + 本体の _detect_score_reset (通常の generate() 経路) を各ターゲット
# 周辺140秒だけ流し、[reset] 行を集める。7点を並列実行する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p logs/split_probes_2026-08-21

run_probe() {
  TAG="$1"; START="$2"; END="$3"
  LOG="logs/split_probes_2026-08-21/probe_${TAG}.log"
  { date; } > "$LOG"
  PYTHONPATH=. ./venv/bin/python -u -m scripts.visualize_advantage_overlay \
    --video data/frames/video_zenchi_c0BQoMJwwQU.mp4 \
    --start-sec "$START" --end-sec "$END" \
    --no-force-in-match --model-dir data/verify/retrain_model62_2026-08-21 \
    --no-render --out "data/verify/_split_probe_${TAG}_dummy.mp4" \
    >> "$LOG" 2>&1
  echo "TAG=$TAG done" >> "$LOG"
}

run_probe t877  837.5  977.5 &
run_probe t1755 1715.0 1855.0 &
run_probe t2632 2592.5 2732.5 &
run_probe t3510 3470.0 3610.0 &
run_probe t4387 4347.5 4487.5 &
run_probe t5265 5225.0 5365.0 &
run_probe t6142 6102.5 6242.5 &
wait
echo ALL_PROBES_DONE
