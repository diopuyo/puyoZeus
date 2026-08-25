#!/bin/bash
# デモレビュー#3/#4/#5/#8 検証用: OLD/NEW構成のタイムラインdump比較 (使い捨て、コミット対象外)。
# 3場面 (source 196s / 235-245s / 347s) を短い窓 (前後わずかな余裕+ウォームアップ) だけ処理する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
IN=data/frames/review_demo_2026-08-12.mp4
OUTDIR=data/verify/demo_review_2026-08-13
mkdir -p "$OUTDIR" logs
BASE="--video $IN --early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure --sample-interval 0 --no-render"

run_one() {
  local name="$1"; local start="$2"; local end="$3"; local warmup="$4"; shift 4
  echo "[run] $name start=$start end=$end warmup=$warmup extra=$*"
  nice -n 19 ./venv/bin/python -u -m scripts.visualize_advantage_overlay $BASE \
    --start-sec "$start" --end-sec "$end" --warmup-sec "$warmup" "$@" \
    --dump-timeline "$OUTDIR/$name.npz" > "logs/demo_review_scene_$name.log" 2>&1
  echo "[done] $name"
}

# シーンA: デモ34秒 = source196s (#1/#2クラスタの回帰確認も兼ねる)
run_one sceneA_old 192 200 10
run_one sceneA_new 192 200 10 --counter-remaining-time --counter-defender-only

# シーンB: デモ73-83秒 = source235-245s (#3/#4/#5の主戦場)
run_one sceneB_old 231 248 10
run_one sceneB_new 231 248 10 --counter-remaining-time --counter-defender-only

# シーンC: デモ185秒 = source347s (#4の自己矛盾ケース)
run_one sceneC_old 343 352 10
run_one sceneC_new 343 352 10 --counter-remaining-time --counter-defender-only

echo "ALL_SCENE_CHECKS_DONE $(date)"
