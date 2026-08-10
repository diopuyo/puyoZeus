#!/bin/bash
# FINAL6: FINAL5 構成 + 状態機械振動バグ B+C の修正 2 フラグ
#   --enable-ojama-entry-gravity-settle-guard (修正B: GRAVITY_SETTLE 中の
#     OJAMA_FALL 新規発火を禁止)
#   --enable-gravity-settle-reset-on-exit (修正C: GRAVITY_SETTLE 横取り退出時に
#     GravitySettleDetector 内部カウンタをリセット)
# 効果測定 (2026-08-08、dio_vs_ts_m01_clip t=45-75s):
#   GRAVITY_SETTLE 中の視覚由来 誤 OJAMA_FALL 発火 4 件 -> 0 件、
#   2P の実 state 遷移 71 -> 62 件 (-12.7%)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt
BASEFLAGS="--cnn-model $MODEL --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce --enable-asymmetric-recovery-min-frames --recovery-add-min-frames 3 --overlay-cell-stability-frames 14 --hide-ojama-forecast --enable-ojama-entry-gravity-settle-guard --enable-gravity-settle-reset-on-exit"

CMD_A="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition --video $IN --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL6a_all_states_viz.mp4 $BASEFLAGS"
CMD_B="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition --video $IN --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL6b_stable_tsumo_viz.mp4 $BASEFLAGS --overlay-show-states stable,tsumo_fall --overlay-state-debounce-frames 10"

{ echo "[cmd] $CMD_A"; eval "$CMD_A"; } > logs/demo_viz_final6a_2026-08-08.log 2>&1 &
{ echo "[cmd] $CMD_B"; eval "$CMD_B"; } > logs/demo_viz_final6b_2026-08-08.log 2>&1 &
wait
echo "FINAL6_BOTH_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL6*.mp4
