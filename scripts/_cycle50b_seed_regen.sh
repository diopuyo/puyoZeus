#!/bin/bash
# cycle 50b: 末尾 skip 360 frame (= 6 秒) で v86m17 中心の再抽出
# 勝利 telop 「やった」 混入対策。 28 動画全再抽出は時間かかるので、
# 朝のレビュー対象 4 動画 + 主要 4 動画優先。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle50b_seed_regen

mkdir -p data/phase_l/seeds_cycle50b
mkdir -p data/seed_review

# 朝レビュー対象 4 動画優先
PRIORITY_VIDEOS=(v86m17 v52m5 v89m7 v34m13)

for key in "${PRIORITY_VIDEOS[@]}"; do
  input="data/phase_l/cut/${key}_buf15s.mp4"
  if [ ! -f "$input" ]; then continue; fi
  if [ -f "data/phase_l/seeds_cycle50b/${key}/cell.jsonl" ]; then continue; fi
  log="logs/cycle50b_seed_regen/seed_${key}.log"
  mkdir -p "$(dirname "$log")"
  run_item seed "$key" \
    ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
      --video "$input" \
      --video-id "$key" \
      --out-root "data/phase_l/seeds_cycle50b" \
      --max-per-color 1500 \
      --max-empty 500
done

# PNG 再生成
for key in "${PRIORITY_VIDEOS[@]}"; do
  d="data/phase_l/seeds_cycle50b/${key}"
  if [ ! -f "${d}/cell.jsonl" ]; then continue; fi
  ./venv/bin/python -m scripts.visualize_seed_samples \
    --seed-root "$d" \
    --output "data/seed_review/cycle50b_${key}.png" \
    --per-color 30 \
    > /dev/null 2>&1
done

# diff 合成 (= cycle50 vs cycle50b)
for key in "${PRIORITY_VIDEOS[@]}"; do
  before="data/seed_review/cycle50_${key}.png"
  after="data/seed_review/cycle50b_${key}.png"
  if [ -f "$before" ] && [ -f "$after" ]; then
    ./venv/bin/python -m scripts.compose_seed_diff \
      --before "$before" --after "$after" \
      --output "data/seed_review/cycle50b_diff_${key}.png" \
      --label "${key} (= cycle 50 → 50b 末尾 skip 拡張)" \
      > /dev/null 2>&1
  fi
done

# S1 audit (= cycle50b 結果)
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_seed_quality \
  --seed-root data/phase_l/seeds_cycle50b \
  --report-out data/verify/seed_quality_cycle50b.json \
  > logs/cycle50b_seed_regen/s1_audit.log 2>&1

finalize_health 0
