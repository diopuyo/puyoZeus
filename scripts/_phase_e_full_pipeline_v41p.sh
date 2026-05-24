#!/bin/bash
# DL 完了後に v41+ について全パイプラインを通す (並列版):
#   1. count_match_v4 (xargs -P 8)
#   2. detect_match_winners (xargs -P 8)
#   3. phase_e_collect_indicator_dataset (workers=8, fps=3)
#   4. shard 統合 (v01-v??.csv 生成)
#   5. learn_weights_v3 + phase_e_learn_phase_aware
#   6. ダッシュボード再生成
#
# 利用例:
#   bash scripts/_phase_e_full_pipeline_v41p.sh

cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
LOG=data/phase_e_full_pipeline.log
> "$LOG"

# 1. v41 以降の動画を検出 (video_41.mp4 から始まる連番)
VIDS=()
for v in $(seq 41 99); do
    VV=$(printf "%02d" "$v")
    if [ -f "data/frames/video_${VV}.mp4" ]; then
        VIDS+=("$v")
    fi
done
if [ ${#VIDS[@]} -eq 0 ]; then
    echo "no v41+ videos found" | tee -a "$LOG"
    exit 1
fi
echo "[target] v41+ videos: ${VIDS[*]}" | tee -a "$LOG"

# 2. count_match_v4 (8 並列、CPU bound)
echo "=== count_match (parallel 8) ===" | tee -a "$LOG"
export -f run_count_match 2>/dev/null
printf "%s\n" "${VIDS[@]}" | xargs -P 8 -I{} bash -c '
v=$1
VV=$(printf "%02d" "$v")
echo "--- count_match v${VV} ---"
./venv/bin/python scripts/count_match_v4.py \
    --video "data/frames/video_${VV}.mp4" \
    --out-root data/verify/match_boundaries_v5 \
    --interval 1 --confirm 3 \
    --min-duration 20 --max-duration 220 2>&1 | tail -3
' _ {} 2>&1 | tee -a "$LOG"

# 3. detect_match_winners (8 並列)
echo "=== detect_winners (parallel 8) ===" | tee -a "$LOG"
printf "%s\n" "${VIDS[@]}" | xargs -P 8 -I{} bash -c '
v=$1
VV=$(printf "%02d" "$v")
matches="data/verify/match_boundaries_v5/video_${VV}/matches.tsv"
n=$(awk "NR>1" "$matches" 2>/dev/null | wc -l)
if [ "$n" -lt 1 ]; then
    echo "[skip] v${VV}: 0 matches"
    exit 0
fi
echo "--- detect_winners v${VV} (${n} matches) ---"
./venv/bin/python scripts/detect_match_winners.py \
    --video "data/frames/video_${VV}.mp4" \
    --matches-tsv "$matches" \
    --out "data/verify/match_winners_v${VV}.tsv" 2>&1 | tail -3
' _ {} 2>&1 | tee -a "$LOG"

# 4. phase_e_collect_indicator_dataset (workers=8, fps=3 で高速化)
VIDS_ARG=$(IFS=, ; echo "${VIDS[*]}")
echo "=== collect (videos=${VIDS_ARG}, workers=8, fps=3) ===" | tee -a "$LOG"
./venv/bin/python -m scripts.phase_e_collect_indicator_dataset \
    --videos "${VIDS_ARG}" --max-matches 0 --fps 3 --workers 8 \
    --out-csv "data/training/match_features_phase_e_v41p_only.csv" 2>&1 | tee -a "$LOG"

# 5. shard 統合 → v01-vXX.csv
echo "=== merge shards ===" | tee -a "$LOG"
LAST_V="${VIDS[-1]}"
LAST_VV=$(printf "%02d" "$LAST_V")
MERGED="data/training/match_features_phase_e_v01-${LAST_VV}.csv"
./venv/bin/python -c "
import csv
from pathlib import Path
shard_dir = Path('data/training/phase_e_shards')
out = Path('${MERGED}')
all_rows = []
fieldnames = None
for shard in sorted(shard_dir.glob('shard_v*.csv')):
    with shard.open(encoding='utf-8') as f:
        r = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = list(r.fieldnames)
        for row in r:
            all_rows.append(row)
with out.open('w', encoding='utf-8', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_rows)
print(f'merged {len(all_rows)} rows -> {out}')
" | tee -a "$LOG"

# 6. 重み学習
echo "=== learn weights ===" | tee -a "$LOG"
./venv/bin/python -m scripts.learn_weights_v3 \
    --csv "${MERGED}" \
    --out "data/verify/learned_weights_v3_phase_e_v01-${LAST_VV}.json" 2>&1 | tail -10 | tee -a "$LOG"
./venv/bin/python -m scripts.phase_e_learn_phase_aware \
    --csv "${MERGED}" \
    --out "data/verify/learned_weights_phase_e_phase_aware_v01-${LAST_VV}.json" 2>&1 | grep -v FutureWarning | grep -v warnings.warn | grep -v deprecated | tail -15 | tee -a "$LOG"

# 7. ダッシュボード再生成 (拡張版)
echo "=== dashboard ===" | tee -a "$LOG"
./venv/bin/python -m scripts.phase_e_dashboard \
    --csv "${MERGED}" \
    --learn-json "data/verify/learned_weights_phase_e_phase_aware_v01-${LAST_VV}.json" \
    --out "data/verify/phase_e_dashboard_v01-${LAST_VV}.md" 2>&1 | tee -a "$LOG"

echo "=== complete ===" | tee -a "$LOG"
