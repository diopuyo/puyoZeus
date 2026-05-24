#!/bin/bash
# cycle 51: ojama 採取 dry-run (= 改修 1 OjamaShapeGate 効果検証)
# v89m7 (= ojama 出現の多い動画) で --include-ojama 付き seed 抽出
# 朝 OK 判定なら cycle 51 として本格採取に移行
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

source scripts/_lib_health.sh
init_health cycle51_ojama_dryrun

# v89m7 で ojama 含み採取 (= 既存 v89m7 が無汚染確認済の 1 動画)
PYTHONPATH=. ./venv/bin/python -m scripts.extract_hsv_seed_dataset \
  --video data/phase_l/cut/v89m7_buf15s.mp4 \
  --video-id v89m7_ojama_dryrun \
  --out-root data/cycle51_ojama_dryrun \
  --max-per-color 500 \
  --max-empty 200 \
  --include-ojama \
  > logs/cycle51_ojama_dryrun.log 2>&1

# seed PNG 生成 (= 朝のレビュー用)
./venv/bin/python -m scripts.visualize_seed_samples \
  --seed-root data/cycle51_ojama_dryrun/v89m7_ojama_dryrun \
  --output data/seed_review/cycle51_ojama_v89m7.png \
  --per-color 30 \
  > /dev/null 2>&1

# S1 audit
PYTHONPATH=. ./venv/bin/python -m scripts.evaluate_seed_quality \
  --seed-root data/cycle51_ojama_dryrun \
  --report-out data/verify/seed_quality_cycle51_ojama_dryrun.json \
  > /dev/null 2>&1

finalize_health 0
