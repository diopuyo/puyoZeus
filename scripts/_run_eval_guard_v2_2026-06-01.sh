#!/bin/bash
# guard 案B (commit 4d97e80 + 00b93f0、両呼出経路ガード、UNKNOWN留保) の eval。
# 案Bが color->empty 副作用を除去したか検証 + 採用候補の最終状態。
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

# guardB単独 (vs baseline clean比較、案Bがcolor->empty副作用を除去したか検証)
run_cfg guardB --infer-empty-guard
# t2yield + guardB (constraint ON)
run_cfg t2_guardB --t2-highconf-yield --infer-empty-guard
# t2yield + guardB + nocfill (最終候補)
run_cfg t2_guardB_nocfill --t2-highconf-yield --infer-empty-guard --no-constraint-fill

echo "[eval] guardB全構成完了: $(date)"
