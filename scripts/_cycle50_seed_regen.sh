#!/bin/bash
# cycle 50: 改修済 seed pipeline で 22 動画再抽出
# 改修 2 (両側 STABLE) + 改修 3 (色別 H filter) + 改修 4 (effect recovery)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle50_seed_regen

mkdir -p data/phase_l/seeds_cycle50
mkdir -p data/seed_review

MAX_PARALLEL=3
pids=()
running=0

for d in data/phase_l/cut/v*m*_buf15s.mp4; do
  key=$(basename "$d" _buf15s.mp4)
  if [ -f "data/phase_l/seeds_cycle50/${key}/cell.jsonl" ]; then
    echo "[skip] $key (already done)"
    continue
  fi
  log="logs/cycle50_seed_regen/seed_${key}.log"
  mkdir -p "$(dirname "$log")"
  (
    PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
      --video "$d" \
      --video-id "$key" \
      --out-root "data/phase_l/seeds_cycle50" \
      --max-per-color 1500 \
      --max-empty 500 \
      > "$log" 2>&1
    rc=$?
    echo "{\"step\":\"seed\",\"item\":\"$key\",\"rc\":$rc,\"ts\":$(date +%s)}" \
      >> "logs/cycle50_seed_regen/_status.jsonl"
  ) &
  pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL ]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
    ((running--)) || true
  fi
done
wait

# 並列で 22 PNG 再生成
for d in data/phase_l/seeds_cycle50/v*m*/; do
  key=$(basename "$d")
  if [ ! -f "${d}cell.jsonl" ]; then continue; fi
  ./venv/bin/python -m scripts.visualize_seed_samples \
    --seed-root "$d" \
    --output "data/seed_review/cycle50_${key}.png" \
    --per-color 30 \
    > /dev/null 2>&1 &
done
wait

# S1 audit
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_seed_quality \
  --seed-root data/phase_l/seeds_cycle50 \
  --report-out data/verify/seed_quality_cycle50.json \
  > logs/cycle50_seed_regen/s1_audit.log 2>&1

finalize_health 0
