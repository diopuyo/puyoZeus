#!/bin/bash
# 3試合連続の判定付きデモ動画 (2026-08-12 user依頼、壁打ち成果の初回確認用)。
# 前提:
#   1. data/verify/npz_light_smoke_2026-08-12/labeled_win_light63.csv が存在
#      (scripts/build_labeled_win_from_npz.py --profile light の63本連結、center_bulge入り)
#   2. scripts/visualize_advantage_overlay.py の TRAIN_CSV_PATH が上記CSVを指す
#      (暫定切替済み。148フル版ができたら差し替える)
# 構成 = production_config の採用フラグから --platt-calibration を除外した素の出力
#   (旧モデル向け較正を新データ学習モデルに当てるのは技術的に不正のため。
#    platt撤回自体は user 判断待ち、8/11からの継続論点)
# 入力 = data/frames/review_demo_2026-08-12.mp4 (video_100 の退避コピー。
#   学習63本に不使用・過去レビュー/デモにも未使用の新規動画 = ホールドアウト)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/review_demo_2026-08-12.mp4
OUTDIR=data/verify/demo_3match_2026-08-12
mkdir -p $OUTDIR logs
# --end-sec は第1引数で調整可 (既定900秒=最初の~3試合を想定。0=全編)
ENDSEC=${1:-900}
CMD="nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay \
  --video $IN \
  --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
  --sample-interval 0 --counter-reach --layout panel --show-recognition \
  --end-sec $ENDSEC \
  --out $OUTDIR/demo_3match_endsec${ENDSEC}.mp4"
echo "[cmd] $CMD"
eval "$CMD"
echo "DEMO_3MATCH_DONE $(date)"
ls -lh $OUTDIR/
