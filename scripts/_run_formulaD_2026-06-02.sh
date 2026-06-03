#!/bin/bash
# 機能D (掛け算式検知 CHAIN 早期発火) の A/B 検証。
# baseline(採用スタック=default) vs 機能D ON(--chain-formula-detection)。
# 採否軸: ①v70 1P連鎖突入の早期化(viz/board_log) ②non_stable悪化なし ③corruption悪化なし。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"
OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"
MASTER="${LOGDIR}/master_formulaD.log"
echo "[start] formulaD A/B $(date)" > "${MASTER}"

# baseline eval (機能D OFF = 現状default)
( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
    --output "${OUTDIR}/corruption_baseline_formulaD_2026-06-02.json" \
    > "${LOGDIR}/eval_baseline_formulaD.log" 2>&1 ; echo "[eval] baseline done $(date)" >> "${MASTER}" ) &
E1=$!

# 機能D ON eval
( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
    --output "${OUTDIR}/corruption_formulaD_2026-06-02.json" --chain-formula-detection \
    > "${LOGDIR}/eval_formulaD.log" 2>&1 ; echo "[eval] formulaD done $(date)" >> "${MASTER}" ) &
E2=$!

# viz: v70_match02 (1P連鎖ラグ本体) baseline vs 機能D ON、board_log ペア保持
( PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
    --video "${VDIR}/v70/v70_match02.mp4" --output "${VIZDIR}/v70_match02_baseline_formulaD_2026-06-02.mp4" \
    --dump-board-log-detailed "${VIZDIR}/v70_match02_baseline_formulaD_2026-06-02.jsonl" \
    > "${LOGDIR}/viz_v70m02_baseline.log" 2>&1
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
    --video "${VDIR}/v70/v70_match02.mp4" --output "${VIZDIR}/v70_match02_formulaD_2026-06-02.mp4" --chain-formula-detection \
    --dump-board-log-detailed "${VIZDIR}/v70_match02_formulaD_2026-06-02.jsonl" \
    > "${LOGDIR}/viz_v70m02_formulaD.log" 2>&1
  PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
    --video "${VDIR}/v70/v70_match01.mp4" --output "${VIZDIR}/v70_match01_formulaD_2026-06-02.mp4" --chain-formula-detection \
    --dump-board-log-detailed "${VIZDIR}/v70_match01_formulaD_2026-06-02.jsonl" \
    > "${LOGDIR}/viz_v70m01_formulaD.log" 2>&1
  echo "[viz] v70 done $(date)" >> "${MASTER}" ) &
Vp=$!

wait $E1 $E2 $Vp
echo "[done] formulaD A/B 全完了 $(date)" >> "${MASTER}"
