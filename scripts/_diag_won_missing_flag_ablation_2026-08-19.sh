#!/bin/bash
# won欠損の境界フラグ切り分けA/B (2026-08-19)
# c109 の 1430-1930s (実試合約9試合、新方式で断片化クラスタ2箇所を含む区間) を
# フラグ構成を変えて収集し、game断片化とwon欠損の変化を見る。
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

BASE="--enable-effect-gate --enable-burst-guard-v2 --enable-transition-merge-guard --burst-gate-open-threshold 0.954 --enable-hidden-row-burst-guard --enable-match-transition-debounce --enable-ojama-fall-placement-override --enable-patch-fp-hsv-guard --enable-floating-gap-restore --enable-landing-color-guard --enable-override-color-guard --enable-ojama-column-stack-fix --enable-next-history-starvation-fix --enable-ojama-cnn-override-warmup --enable-ojama-write-accounting-guard --enable-ojama-fall-color-swap-guard --enable-chain-tracker --enable-stable-persistence-gate --enable-winner-panel-crosscheck --enable-move-segmented-recording --enable-physics-persistence-filter"
B3="--enable-match-end-persist-override --enable-post-match-lockdown-latch --enable-result-screen-hardening"
COMMON="--with-next --enable-phantom-board-guard --start-sec 1430 --max-sec 500 --sample-interval 0"
VIDEO=data/frames/video_c109.mp4
OUT=logs/_diag_flag_ablation_2026-08-19
mkdir -p "$OUT"

run() {
  name="$1"; shift
  echo "=== [$name] start $(date +%H:%M:%S) ==="
  PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_lean_1t \
    --video "$VIDEO" --out-npz "$OUT/$name.npz" $COMMON "$@" \
    > "$OUT/$name.log" 2>&1
  echo "=== [$name] done rc=$? $(date +%H:%M:%S) ==="
}

# A: 本番フル構成 (再現対照)
run A_full $BASE $B3 --enable-boundary-multisignal
# B: フルからlockdown-latchのみOFF
run B_no_lockdown $BASE --enable-match-end-persist-override --enable-result-screen-hardening --enable-boundary-multisignal
# C: フルからmultisignalのみOFF (境界=score-reset単独に戻す)
run C_no_multisignal $BASE $B3
# D: 境界3フラグ全OFF + multisignal ON (multisignal単独の寄与)
run D_no_b3 $BASE --enable-boundary-multisignal
# E: 旧方式相当 (境界系4種全OFF)
run E_legacy_boundary $BASE
echo "ALL DONE"
