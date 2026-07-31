#!/bin/bash
# CSV 行数増加を60秒測って実スループット(行/分)を出す
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
FILES="v30_gap v30_mid v31_gap v31_mid v32 v32_mid v33_mid v34_gap"
declare -A before
for f in $FILES; do
  before[$f]=$(wc -l < "data/indicators_v2/study/$f.csv" 2>/dev/null || echo 0)
done
sleep 60
total=0
for f in $FILES; do
  now=$(wc -l < "data/indicators_v2/study/$f.csv" 2>/dev/null || echo 0)
  d=$((now - ${before[$f]}))
  total=$((total + d))
  echo "$f: ${before[$f]} -> $now (+$d 行/分)"
done
echo "合計: +$total 行/分 (12ジョブ)"
