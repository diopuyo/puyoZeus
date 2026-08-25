#!/bin/bash
# 30先2セット動画の全8区間を「修正① (kill_override 連鎖完走後是正) ON」で
# 再走査する (2026-08-22、user承認・「全体を待つ」判断)。
#
# _render_zenchi_8seg_2026-08-21.sh と同じ区間分割・同じ本番採用フラグ構成
# (production_config.advantage_overlay_flags() が単一情報源) を使うが、
# 以下の2点だけ異なる:
#   1. render=False (--no-render 相当、動画書き出しなし) — dump (npz) だけを
#      得れば scripts/_diag_kill_raw_display_conflict_2026-08-22.py --dir で
#      「修正後は112エピソードから何件に減ったか」を測れるため、
#      長時間レンダ (動画エンコード) を避ける (user指示「長時間レンダは
#      起動しない」に抵触しない構成)。
#   2. --kill-override-chain-completion を追加 (修正①、既定OFF・
#      未登録フラグなので手動追加。他の修正②④は本測定の対象外のため
#      あえて付けない = 一度に1変数だけ変える対照実験)。
#
# 出力は元の本番dump (data/verify/zenchi_render_2026-08-21) を一切上書き
# しない別ディレクトリに書く (before/after 比較を保持するため)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

# BLAS/OpenMP 系のスレッドを1に固定 (_render_zenchi_8seg_2026-08-21.sh と
# 同じ教訓、過剰スレッド生成によるスループット崩壊の防止)。
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export CV_NUM_THREADS=1

VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
MODEL=data/verify/retrain_model62_2026-08-21
OUTDIR=data/verify/zenchi_render_kill_override_fix_v2_2026-08-22
LOGDIR=logs/zenchi_rescan_kill_override_fix_v2_2026-08-22
WARMUP=30

mkdir -p "$OUTDIR" "$LOGDIR"

# _render_zenchi_8seg_2026-08-21.sh と完全同一の区間境界 (試合開始、目視
# ±0秒精度確認済み)。before/after を同じ切り口で比較するため変更しない。
BOUNDS=(0 893.7 1738.3 2637.3 3626.0 4379.5 5255.6 6131.6 7033.6)
N=$((${#BOUNDS[@]} - 1))

ADOPTED=$(PYTHONPATH=. ./venv/bin/python -c \
  "import src.production_config as pc; print(pc.advantage_overlay_flags())")
if [ -z "$ADOPTED" ]; then
  echo "=== 中止: advantage_overlay_flags() が空文字を返した (production_config.py 側の異常) ==="
  exit 1
fi

# 修正①のみ追加 (--kill-override-chain-completion、単一変数の対照実験)。
FLAGS="--layout panel --panel-subtitle-h 0 --no-force-in-match --no-render \
--model-dir $MODEL --warmup-sec $WARMUP \
--resolved-exchange-eval --resolved-decisive-amplify --resolved-live-defender \
--kill-override-chain-completion \
$ADOPTED"

echo "=== 30先動画 修正①(改良②累積)検証用 全編再走査 start $(date +%F_%T) ==="
echo "[単一情報源] $ADOPTED"
echo "[構成] $FLAGS"
echo "[区間] $N 区間、並列 $N、暖機 ${WARMUP}秒 (render無し=動画書き出しなし)"
cat /proc/loadavg

for i in $(seq 1 "$N"); do
  S=${BOUNDS[$((i - 1))]}
  E=${BOUNDS[$i]}
  OUT="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.mp4"
  DUMP="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.npz"
  LOG="$LOGDIR/seg$(printf '%02d' "$i").log"
  echo "[起動] 区間$i  $S 〜 $E 秒 (長さ $(echo "$E - $S" | bc)s) -> $DUMP"
  (
    echo "=== seg$i start $(date +%F_%T) 範囲 $S〜$E 暖機 ${WARMUP}s ==="
    echo "[flags] $FLAGS"
    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_advantage_overlay \
      --video "$VIDEO" --start-sec "$S" --end-sec "$E" \
      $FLAGS --dump-timeline "$DUMP" --out "$OUT"
    echo "=== seg$i done rc=$? $(date +%F_%T) ==="
  ) > "$LOG" 2>&1 &
done

echo "--- 全${N}区間を起動、完了待ち ---"
wait
echo "=== 全区間完了 $(date +%F_%T) ==="
ls -l "$OUTDIR"
echo "--- 各区間の終了コード ---"
grep -h "done rc=" "$LOGDIR"/seg*.log
