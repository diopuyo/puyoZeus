#!/bin/bash
# 各 evaluation_videos に対応する match の start_sec を表示
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

# evaluation_videos のファイル名から (video_id, match_idx) を抽出
declare -A MATCH_MAP=(
  ["v29m2"]="29 2"
  ["v40m7"]="40 7"
  ["v51m2"]="51 2"
  ["v57m2"]="57 2"
  ["v70m2"]="70 2"
  ["v89m3"]="89 3"
  ["v95m15"]="95 15"
  ["v97m11"]="97 11"
)

for key in v29m2 v40m7 v51m2 v57m2 v70m2 v89m3 v95m15 v97m11; do
  info=${MATCH_MAP[$key]}
  vid=$(echo $info | cut -d' ' -f1)
  mid=$(echo $info | cut -d' ' -f2)
  # match_boundaries_v4 をデフォルトで使う、 なければ v5
  tsv=""
  for ver in v5 v4; do
    candidate=data/verify/match_boundaries_${ver}/video_${vid}/matches.tsv
    if [ -f "$candidate" ]; then
      tsv=$candidate
      break
    fi
  done
  if [ -z "$tsv" ]; then
    echo "$key: tsv not found"
    continue
  fi
  # match_idx 行 (= idx, start_sec, end_sec, duration_sec)
  line=$(awk -F'\t' -v idx=$mid 'NR>1 && $1==idx {print}' "$tsv")
  echo "$key: tsv=$tsv match $mid: $line"
done
