#!/bin/bash
# FINAL10: FINAL9 + --chain-formula-simulate-verify (連鎖数を固定1でなく実測値にする) + 連鎖数表示
# あってほしい」、 2026-08-08)。
#   --overlay-chain-hold-until-end : GRAVITY_SETTLE を chain として表示し、
#       物理推論側の ChainEvent.end_sec (連鎖終了予測時刻) までは途中の
#       誤遷移 (ojama_fall 等) が起きても chain 表示を維持する。
#       : chain 表示中に連鎖数を併記 (例 9renza)。
# 実測 (dio_vs_ts_m01_clip t=45-75s): 連鎖中の離脱 18 回は全て
# GRAVITY_SETTLE 起点、 うち 6 回はそこから OJAMA_FALL へ抜けていた。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/verify/youtube_demo_2026-08-07/dio_vs_ts_m01_clip.mp4
MODEL=models/cnn_finetune_olRyxDGacbg_demo_v3_2026-08-07.pt
BASEFLAGS="--cnn-model $MODEL --enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce --enable-asymmetric-recovery-min-frames --recovery-add-min-frames 3 --overlay-cell-stability-frames 14 --chain-formula-simulate-verify --overlay-show-chain-count --hide-ojama-forecast --enable-ojama-entry-gravity-settle-guard --enable-gravity-settle-reset-on-exit --overlay-chain-hold-until-end"

CMD_A="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition --video $IN --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL10a_all_states_viz.mp4 $BASEFLAGS"
CMD_B="nice -n 19 ./venv/bin/python -u -m scripts.visualize_recognition --video $IN --output data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL10b_stable_tsumo_viz.mp4 $BASEFLAGS --overlay-show-states stable,tsumo_fall --overlay-state-debounce-frames 10"

{ echo "[cmd] $CMD_A"; eval "$CMD_A"; } > logs/demo_viz_final10a_2026-08-08.log 2>&1 &
{ echo "[cmd] $CMD_B"; eval "$CMD_B"; } > logs/demo_viz_final10b_2026-08-08.log 2>&1 &
wait
echo "FINAL10_BOTH_DONE $(date)"
ls -lh data/verify/youtube_demo_2026-08-07/dio_vs_ts_FINAL10*.mp4
