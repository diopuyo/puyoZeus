#!/bin/bash
# 案Y-4 deferred consensus (着地色のHSV-first確定+consensus投票) のA/B検証。
# deferredは enable_hsv_classify_fallback=True が前提(併用必須)。
# baseline(両OFF=現default)は既存 corruption_formulaD_2026-06-02.json を流用。
# config2 = --hsv-classify-fallback 単独 (fallback ON / deferred OFF)
# config3 = --hsv-classify-fallback --hsv-deferred-consensus (本修正)
# config3 vs default=総改善、config3 vs config2=deferred純効果。
# viz は corruption濃い v89m01/v70m2 を config3 で生成(before=既存default board_log)。
set -u
PROJ_DIR="/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer"; cd "${PROJ_DIR}"
V="v29_match01,v29_match02,v40_match01,v40_match02,v51_match01,v51_match02,v57_match01,v57_match02,v70_match01,v70_match02,v89_match01,v89_match02,v95_match01,v95_match02,v97_match01,v97_match02"
H="v29_match01,v29_match02,v40_match01,v40_match02,v89_match01,v89_match02"
VDIR="data/match_clips"; SI="0.03333333"
OUTDIR="data/verify/stable_cell_acc"; VIZDIR="data/verify/viz"; LOGDIR="logs/fix_v70_eval"
mkdir -p "${OUTDIR}" "${VIZDIR}" "${LOGDIR}"
MASTER="${LOGDIR}/master_deferred.log"
echo "[start] deferred A/B $(date)" > "${MASTER}"

# config2: fallback単独 (deferred OFF)
( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
    --hsv-classify-fallback \
    --output "${OUTDIR}/corruption_hsvfb_only_2026-06-03.json" \
    > "${LOGDIR}/eval_hsvfb_only.log" 2>&1 ; echo "[eval] config2 fallback-only done $(date)" >> "${MASTER}" ) &
E1=$!

# config3: fallback + deferred consensus (本修正)
( PYTHONPATH=. venv/bin/python scripts/measure_stable_cell_acc.py \
    --videos "${V}" --holdout "${H}" --video-dir "${VDIR}" --sample-interval "${SI}" --workers 6 \
    --hsv-classify-fallback --hsv-deferred-consensus \
    --output "${OUTDIR}/corruption_deferred_2026-06-03.json" \
    > "${LOGDIR}/eval_deferred.log" 2>&1 ; echo "[eval] config3 deferred done $(date)" >> "${MASTER}" ) &
E2=$!

wait $E1 $E2
echo "[eval] 両eval完了、viz開始 $(date)" >> "${MASTER}"

# viz: config3 で corruption濃い動画 (before=既存default board_log と比較)
( for p in "v89_match01:v89" "v70_match02:v70"; do v="${p%%:*}"; d="${p##*:}"
    PYTHONPATH=. venv/bin/python scripts/visualize_recognition.py \
      --video "${VDIR}/${d}/${v}.mp4" --output "${VIZDIR}/${v}_deferred_2026-06-03.mp4" \
      --hsv-classify-fallback --hsv-deferred-consensus \
      --dump-board-log-detailed "${VIZDIR}/${v}_deferred_2026-06-03.jsonl" \
      > "${LOGDIR}/viz_${v}_deferred.log" 2>&1
  done ; echo "[viz] config3 done $(date)" >> "${MASTER}" ) &
Vp=$!
wait $Vp
echo "[done] deferred A/B 全完了 $(date)" >> "${MASTER}"
