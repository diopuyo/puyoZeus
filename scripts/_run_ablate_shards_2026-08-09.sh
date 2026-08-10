#!/bin/bash
# アブレーション1 本走行: 148 npz を14シャードに分けて並列 extract
# 完了後に per_row_values.tsv を結合し auc モードを1回実行する
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
SRC=data/indicators_v2/boards_lean_phase_l_2026-08-07
OUT=data/verify/ablate_exact_k12_2026-08-09
NSHARD=14
mkdir -p "$OUT" logs
# WSLネイティブ側にシャードdirを作る (/mnt/c 上のsymlinkは遅い・不安定なためコピーでなくlinkはtmpfs回避で$HOME)
SHROOT="$HOME/ablate_shards_2026-08-09"
rm -rf "$SHROOT"; mkdir -p "$SHROOT"
i=0
for f in "$SRC"/*.npz; do
  d="$SHROOT/shard_$((i % NSHARD))"
  mkdir -p "$d"
  ln -s "$(pwd)/$f" "$d/$(basename "$f")"
  i=$((i+1))
done
echo "[shards] $i files -> $NSHARD shards"
pids=()
for s in $(seq 0 $((NSHARD-1))); do
  mkdir -p "$OUT/shard_$s"
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.     ./venv/bin/python -u scripts/_ablate_exact_k12_2026-08-09.py     --mode extract --npz-dir "$SHROOT/shard_$s" --out-dir "$OUT/shard_$s"     > "logs/ablate_shard_${s}_2026-08-09.log" 2>&1 &
  pids+=($!)
done
echo "[launch] ${#pids[@]} workers"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "[join] failed=$fail"
# 結合 (ヘッダは先頭シャードのみ)
head -1 "$OUT/shard_0/per_row_values.tsv" > "$OUT/per_row_values.tsv"
for s in $(seq 0 $((NSHARD-1))); do
  tail -n +2 "$OUT/shard_$s/per_row_values.tsv" >> "$OUT/per_row_values.tsv" 2>/dev/null
done
wc -l "$OUT/per_row_values.tsv"
# AUC 比較
PYTHONPATH=. ./venv/bin/python -u scripts/_ablate_exact_k12_2026-08-09.py   --mode auc --out-dir "$OUT"
echo "[done] $(date +%H:%M:%S)"
