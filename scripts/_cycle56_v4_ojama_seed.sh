#!/bin/bash
# cycle 56_v4: ojama seed を含む 7 クラス seed 採取.
# cycle 50 final (= 5 色 + EMPTY、 ojama 0 件) では cycle 56_v2 で ojama 退行発生。
# --include-ojama で ojama を含めて再採取し、 cycle 56_v4 で真 fine-tune して
# ojama 認識保持 + 5 色微改善を両立目指す。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle56_v4_ojama_seed

mkdir -p data/phase_l/seeds_cycle56_ojama
mkdir -p logs/cycle56_v4_ojama_seed

MAX_PARALLEL=3
pids=()
running=0

for d in data/phase_l/cut/v*m*_buf15s.mp4; do
  key=$(basename "$d" _buf15s.mp4)
  if [ -f "data/phase_l/seeds_cycle56_ojama/${key}/cell.jsonl" ]; then
    echo "[skip] $key (already done)"
    continue
  fi
  log="logs/cycle56_v4_ojama_seed/seed_${key}.log"
  (
    PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
      --video "$d" \
      --video-id "$key" \
      --out-root "data/phase_l/seeds_cycle56_ojama" \
      --max-per-color 1500 \
      --max-empty 500 \
      --include-ojama \
      > "$log" 2>&1
    rc=$?
    echo "{\"step\":\"seed\",\"item\":\"$key\",\"rc\":$rc,\"ts\":$(date +%s)}" \
      >> "logs/cycle56_v4_ojama_seed/_status.jsonl"
  ) &
  pids+=($!)
  ((running++)) || true
  if [ $running -ge $MAX_PARALLEL ]; then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
    ((running--)) || true
  fi
done

# 残り wait
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "=== seed 採取完了 @ $(date) ==="

# ojama seed カウント集計
PYTHONPATH=. ./venv/bin/python -c "
import json
from pathlib import Path
import collections

root = Path('data/phase_l/seeds_cycle56_ojama')
total_per_label = collections.Counter()
per_video_ojama = {}
for vd in sorted(root.iterdir()):
    if not vd.is_dir():
        continue
    cf = vd / 'cell.jsonl'
    if not cf.is_file():
        continue
    cnts = collections.Counter()
    with cf.open(encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            lab = obj.get('label')
            cnts[lab] += 1
            total_per_label[lab] += 1
    n_oj = cnts.get(9, 0)
    per_video_ojama[vd.name] = (cnts.get(9, 0), sum(cnts.values()))

print('=== seed_count per label (全 27 動画合計) ===')
for lab in sorted(total_per_label.keys()):
    print(f'  label {lab}: {total_per_label[lab]:,}')
print()
print('=== per-video ojama 件数 (上位 10) ===')
for v, (n_oj, n_total) in sorted(per_video_ojama.items(), key=lambda x: -x[1][0])[:10]:
    print(f'  {v}: ojama={n_oj}, total={n_total}')
print()
print(f'=== total ojama: {total_per_label.get(9, 0)} ===')
"

finalize_health 0
