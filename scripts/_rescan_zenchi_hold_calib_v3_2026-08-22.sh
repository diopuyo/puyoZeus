#!/bin/bash
# 30先2セット動画の全8区間を「根治② (CHAIN保持時間の実測較正配線) ON +
# kill_override連鎖完走後是正 ON + 累積 (ChainGenerationAccumulator) OFF」
# で再走査する (2026-08-22、user判断: 累積は対症療法の実測欠陥のため外し、
# 既存の実測較正式 [2.61+1.17×N、src/recognition_pipeline.py:731-736、
# 2026-07-24較正済みだが未配線だった] を配線した構成だけで測る)。
#
# _render_zenchi_8seg_2026-08-21.sh と同じ区間分割・同じ本番採用フラグ構成
# (production_config.advantage_overlay_flags() が単一情報源) を使うが、
# 以下の3点だけ異なる:
#   1. render=False (--no-render 相当、動画書き出しなし)
#   2. --kill-override-chain-completion を追加 (修正①、既定OFF・未登録
#      フラグなので手動追加)
#   3. --chain-hold-base-sec 2.61 --chain-hold-per-step-sec 1.17 を追加
#      (根治②、実測較正式)。--kill-override-chain-gen-accumulate は
#      **付けない** (既定 False = 累積オフ、user判断)。
#
# 出力は既存の本番dump・v1(単発)・v2(累積)を一切上書きしない別ディレクトリに
# 書く (before/after 比較を保持するため)。
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
OUTDIR=data/verify/zenchi_render_hold_calib_v3_2026-08-22
LOGDIR=logs/zenchi_rescan_hold_calib_v3_2026-08-22
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

# 根治② (CHAIN保持時間の実測較正) + 修正① (kill_override連鎖完走後是正)。
# 累積 (--kill-override-chain-gen-accumulate) は付けない = 既定OFF。
FLAGS="--layout panel --panel-subtitle-h 0 --no-force-in-match --no-render \
--model-dir $MODEL --warmup-sec $WARMUP \
--resolved-exchange-eval --resolved-decisive-amplify --resolved-live-defender \
--kill-override-chain-completion \
--chain-hold-base-sec 2.61 --chain-hold-per-step-sec 1.17 \
$ADOPTED"

echo "=== 30先動画 根治②(CHAIN保持時間実測較正)検証用 全編再走査 start $(date +%F_%T) ==="
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
