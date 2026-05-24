#!/bin/bash
# cycle 58 (= 案 A): 38 動画で 7 クラス seed 採取 (= ojama 含む).
# 並列 3 で時間短縮。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

mkdir -p data/phase_l/seeds_cycle58
mkdir -p logs/cycle58_seed

MAX_PARALLEL=3
pids=()
running=0

for d in data/phase_l/cut/v*m*_buf15s.mp4; do
  key=$(basename "$d" _buf15s.mp4)
  if [ -f "data/phase_l/seeds_cycle58/${key}/cell.jsonl" ]; then
    echo "[skip] $key (already done)"
    continue
  fi
  log="logs/cycle58_seed/seed_${key}.log"
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
    echo "[done] $key"
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

echo "=== seed 採取完了 @ $(date) ==="

# ojama 件数集計
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path
import collections
root = Path('data/phase_l/seeds_cycle58')
total_per_label = collections.Counter()
per_video = {}
for vd in sorted(root.iterdir()):
    if not vd.is_dir(): continue
    cf = vd / 'cell.jsonl'
    if not cf.is_file(): continue
    cnts = collections.Counter()
    with cf.open(encoding='utf-8') as f:
        for line in f:
            cnts[json.loads(line).get('label')] += 1
            total_per_label[json.loads(line).get('label')] += 1
    per_video[vd.name] = (cnts.get(9, 0), sum(cnts.values()))
print('=== label 別集計 ===')
for lab in sorted(total_per_label.keys()):
    print(f'  label {lab}: {total_per_label[lab]:,}')
print(f'\\n=== per-video ojama (= 上位 10) ===')
for v, (oj, total) in sorted(per_video.items(), key=lambda x: -x[1][0])[:10]:
    print(f'  {v}: ojama={oj} total={total}')
print(f'\\n=== ojama 合計: {total_per_label.get(9, 0)} ===')
"
