#!/bin/bash
# cycle 58 seed 採取 追加並列 3 (= 既存 3 並列と合わせて 6 並列).
# 既存 script は alphabet 順、 本 script は逆順で重複削減。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/phase_l/seeds_cycle58
mkdir -p logs/cycle58_seed

MAX_PARALLEL=3
pids=()
running=0

# 逆順で処理
for d in $(ls data/phase_l/cut/v*m*_buf15s.mp4 | sort -r); do
  key=$(basename "$d" _buf15s.mp4)
  if [ -f "data/phase_l/seeds_cycle58/${key}/cell.jsonl" ]; then
    continue
  fi
  # 既存 script が dir 作成中なら skip
  if [ -d "data/phase_l/seeds_cycle58/${key}" ]; then
    continue
  fi
  log="logs/cycle58_seed/seed_${key}_extra.log"
  (
    PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
      --video "$d" \
      --video-id "$key" \
      --out-root "data/phase_l/seeds_cycle58" \
      --max-per-color 1500 \
      --max-empty 500 \
      --include-ojama \
      --ojama-no-gate \
      > "$log" 2>&1
    echo "[done extra] $key"
  ) &
  pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL ]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
    ((running--)) || true
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "=== extra seed 採取完了 @ $(date) ==="
