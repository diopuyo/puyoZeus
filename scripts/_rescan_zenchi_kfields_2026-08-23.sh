#!/bin/bash
# 30先2セット動画の全8区間を、現行本番構成 (_render_zenchi_8seg_2026-08-21.sh
# と完全同一のフラグ = --kill-override-chain-completion +
# --enable-slide-exit-min-display-guard + $ADOPTED) で再走査し、
# 「盲点根治①」(kpending_p1/p2・kroom1/kroom2 を dump に記録する改修) 適用後の
# npz を得る (2026-08-23)。
#
# 目的: 既存 dump (data/verify/zenchi_render_2026-08-21) は本番と同じ判定を
# 行っているが、コード側に是正後の値を記録する仕組みが無かった時点で
# 生成されたため、kpending_p1 等のキーを持たない。scripts/_diag_kill_raw_
# display_conflict_2026-08-22.py --compare-raw で「生値ベースの旧ロジック」と
# 「是正後ベースの新ロジック」を同一区間・同一判定条件で比較するには、
# 新コードで再生成した dump が要る。
#
# render=False (--no-render 相当) で動画書き出しを省き、npz だけを得る
# (_rescan_zenchi_kill_override_fix_v2_2026-08-22.sh と同じ設計判断)。
# 出力は既存 dump を一切上書きしない別ディレクトリに書く
# (user指示: 既存の dump は上書きしないこと)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export CV_NUM_THREADS=1

VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
MODEL=data/verify/retrain_model62_2026-08-21
OUTDIR=data/verify/zenchi_render_kfields_2026-08-23
LOGDIR=logs/zenchi_rescan_kfields_2026-08-23
WARMUP=30

mkdir -p "$OUTDIR" "$LOGDIR"

# _render_zenchi_8seg_2026-08-21.sh と完全同一の区間境界 (試合開始)。
BOUNDS=(0 893.7 1738.3 2637.3 3626.0 4379.5 5255.6 6131.6 7033.6)
N=$((${#BOUNDS[@]} - 1))

ADOPTED=$(PYTHONPATH=. ./venv/bin/python -c \
  "import src.production_config as pc; print(pc.advantage_overlay_flags())")
if [ -z "$ADOPTED" ]; then
  echo "=== 中止: advantage_overlay_flags() が空文字を返した (production_config.py 側の異常) ==="
  exit 1
fi

# 現行本番レンダ (_render_zenchi_8seg_2026-08-21.sh) と完全同一の判定構成。
# 唯一の違いは --no-render (動画書き出し無し) のみ。
FLAGS="--layout panel --panel-subtitle-h 0 --no-force-in-match --no-render \
--model-dir $MODEL --warmup-sec $WARMUP \
--resolved-exchange-eval --resolved-decisive-amplify --resolved-live-defender \
--kill-override-chain-completion --enable-slide-exit-min-display-guard \
$ADOPTED"

echo "=== 30先動画 根治①(kpending/kroom記録)検証用 全編再走査 start $(date +%F_%T) ==="
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
  echo "[起動] 区間$i  $S 〜 $E 秒 -> $DUMP"
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
