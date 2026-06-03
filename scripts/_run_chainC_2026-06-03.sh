#!/bin/bash
# 機能C (chain_exit_warmup=CHAIN→STABLE後0.1秒confirmed凍結) の検証。
# baseline = 機能D ON (採用済=現default)。test = 機能D + --chain-exit-warmup。
# baseline eval は既存 corruption_formulaD_2026-06-02.json を流用 (機能D ON, 機能C OFF)。
# 目的: 連鎖中エフェクト残光誤認の chain→stable 境界混入を機能Cで抑止できるか。
# 主検証 = v89(エフェクト濃い)の viz目視 (consensus corruption の fail-silent 盲点のため)。
# 反省: eval/viz 同時大量起動は競合で激遅 → eval workers8 + viz sequential(同時=9コア)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"
OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"
MASTER="${LOGDIR}/master_chainC.log"
echo "[start] chainC (機能D+機能C) $(date)" > "${MASTER}"

# eval: 機能D(default ON) + 機能C(--chain-exit-warmup)
( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 8 \
    --chain-exit-warmup \
    --output "${OUTDIR}/corruption_formulaD_chainC_2026-06-03.json" \
    > "${LOGDIR}/eval_chainC.log" 2>&1 ; echo "[eval] D+C done $(date)" >> "${MASTER}" ) &
E=$!

# viz: v89 (エフェクト濃い) で機能C有無を比較、board_log ペア保持
( for v in v89_match01 v89_match02; do
    # baseline (機能D only, 機能C OFF)
    PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
      --video "${VDIR}/v89/${v}.mp4" --output "${VIZDIR}/${v}_D_2026-06-03.mp4" \
      --dump-board-log-detailed "${VIZDIR}/${v}_D_2026-06-03.jsonl" \
      > "${LOGDIR}/viz_${v}_D.log" 2>&1
    # 機能D + 機能C
    PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
      --video "${VDIR}/v89/${v}.mp4" --output "${VIZDIR}/${v}_DC_2026-06-03.mp4" --chain-exit-warmup \
      --dump-board-log-detailed "${VIZDIR}/${v}_DC_2026-06-03.jsonl" \
      > "${LOGDIR}/viz_${v}_DC.log" 2>&1
  done ; echo "[viz] v89 D/DC done $(date)" >> "${MASTER}" ) &
Vp=$!

wait $E $Vp
echo "[done] chainC 全完了 $(date)" >> "${MASTER}"
