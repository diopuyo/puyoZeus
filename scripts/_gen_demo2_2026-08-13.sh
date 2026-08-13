#!/bin/bash
# 改修デモ2本目 (2026-08-13): デモ1本目 (_gen_demo_fixed_2026-08-13.sh、video_100) と
# 同一の全修正フラグ構成 + 本番認識構成 (自動適用) で、未見動画に対して再生成する。
#
# 選定動画: video_74 (= YouTube 7GzyjbnMJYs、【マスター・3ブロック】light vs スラさん
#   30先、第2回新おいうリーグ plC#34、video_idx=74)。
# 選定根拠: data/phase_e_dl_index.tsv の video_idx=74 行でタイトル「マスター」確認済。
#   data/verify/regen_2026-08-11_manifest.tsv (148収集対象) に video_idx=74 は不在
#   =学習データ未使用。video_100 (デモ1)・c34・物差しc10-c23 とも重複なし。
#
# 区間 (score OCR 粗スキャン scripts/_scan_score_series_video74_2026-08-13.py で特定):
#   1試合目の実開始 t≈237.4s (0-0 stable) → 少し前の t=230 から開始。
#   3試合目の終了 t≈405.7s (score表示が None に落ちる=結果画面遷移) → +5秒強で t=411 まで。
#   (試合境界: t=284/340/406 で 0-0 リセット確認済み、464/450/490/528/614 も後続に検出)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/video_74.mp4
OUTDIR=data/verify/demo_fixed_2026-08-13
mkdir -p $OUTDIR logs
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay \
  --video $IN \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --layout panel --show-recognition \
  --counter-remaining-time --counter-defender-only \
  --stable-majority-window \
  --enable-ojama-fall-placement-override --enable-ojama-fall-entry-hardening \
  --enable-ojama-fall-scoped-exit \
  --start-sec 230 --end-sec 411 \
  --out $OUTDIR/demo2_video74_3match.mp4"
echo "[cmd] $CMD"
eval "$CMD"
echo "DEMO2_DONE $(date)"
ls -lh $OUTDIR/
