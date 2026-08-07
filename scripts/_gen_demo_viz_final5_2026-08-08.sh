#!/bin/bash
# FINAL5: v3モデル × セル安定フィルタ14f × 非対称復旧ゲート(空→色=3, 色→空/色→色=8)
# × (a:全状態表示 / b:stable+ツモ落下のみ) の2本
# 設置確定レイテンシA/B実験 (data/verify/recovery_min_frames_ab_2026-08-08) で
# 非対称3/8が一律短縮より効果大・汚染量小と確認済みの構成を反映したデモ。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt
BASEFLAGS="--cnn-model $MODEL --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce --enable-asymmetric-recovery-min-frames --recovery-add-min-frames 3 --overlay-cell-stability-frames 14"

CMD_A="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition --video $IN --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL5a_all_states_viz.mp4 $BASEFLAGS"
CMD_B="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition --video $IN --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL5b_stable_tsumo_viz.mp4 $BASEFLAGS --overlay-show-states stable,tsumo_fall --overlay-state-debounce-frames 10"

{ echo "[cmd] $CMD_A"; eval "$CMD_A"; } > logs/demo_viz_final5a_2026-08-08.log 2>&1 &
{ echo "[cmd] $CMD_B"; eval "$CMD_B"; } > logs/demo_viz_final5b_2026-08-08.log 2>&1 &
wait
echo "FINAL5_BOTH_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL5*.mp4
