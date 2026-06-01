#!/bin/bash
# infer_placement 空セルガード (commit 814d55f) の eval。
# 新サブカテゴリ検知 (empty_to_color/color_to_color/color_to_empty, commit aad223d) 付き。
# 4 構成を順次実行 (並列上限 3 順守のため逐次)。
set -u

PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"
cd "${PROJ_DIR}"

VIDEOS="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
HOLDOUT="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"
SI="0.03333333"
W="6"
OUTDIR="data/verify/stable_cell_acc"
LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${LOGDIR}"

run_cfg() {
  local name="$1"; shift
  echo "[eval] === ${name} 開始: $(date) ==="
  PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${VIDEOS}" --holdout "${HOLDOUT}" \
    --video-dir "${VDIR}" --sample-interval "${SI}" --workers "${W}" \
    --output "${OUTDIR}/corruption_${name}_2026-06-01.json" \
    "$@" > "${LOGDIR}/eval_${name}_2026-06-01.log" 2>&1
  echo "[eval] === ${name} 完了: $(date) ==="
}

# A: baseline (無フラグ、新サブカテゴリ検知の基準)
run_cfg baseline
# B: guard のみ
run_cfg guard --infer-empty-guard
# C: t2yield + guard (constraint ON)
run_cfg t2_guard --t2-highconf-yield --infer-empty-guard
# D: t2yield + guard + nocfill
run_cfg t2_guard_nocfill --t2-highconf-yield --infer-empty-guard --no-constraint-fill

echo "[eval] 全構成完了: $(date)"
