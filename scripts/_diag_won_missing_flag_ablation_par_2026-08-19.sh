#!/bin/bash
# A/B残り4構成 (B/C/D/E) を並列起動する (A_fullは既走行、2026-08-19)
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

BASE="--enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce --enable-ojama-fall-placement-override --enable-patch-fp-hsv-guard --enable-floating-gap-restore --enable-landing-color-guard --enable-override-color-guard --enable-ojama-column-stack-fix --enable-next-history-starvation-fix --enable-ojama-cnn-override-warmup --enable-ojama-write-accounting-guard --enable-ojama-fall-color-swap-guard --enable-chain-tracker --enable-stable-persistence-gate --enable-winner-panel-crosscheck --enable-move-segmented-recording --enable-physics-persistence-filter"
B3="--enable-match-end-persist-override --enable-post-match-lockdown-latch --enable-result-screen-hardening"
COMMON="--with-next --enable-phantom-board-guard --start-sec 1430 --max-sec 500 --sample-interval 0"
VIDEO=data/frames/video_c109.mp4
OUT=logs/_diag_flag_ablation_2026-08-19

run_bg() {
  name="$1"; shift
  setsid -f bash -c "cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer && PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_lean_1t --video $VIDEO --out-npz $OUT/$name.npz $COMMON $* > $OUT/$name.log 2>&1 < /dev/null"
  echo "launched $name"
}

run_bg B_no_lockdown $BASE --enable-match-end-persist-override --enable-result-screen-hardening --enable-boundary-multisignal
run_bg C_no_multisignal $BASE $B3
run_bg D_no_b3 $BASE --enable-boundary-multisignal
run_bg E_legacy_boundary $BASE
