#!/bin/bash
# 30先2セット動画の全8区間を「修正①(kill_override連鎖完走後是正)+
# 修正③(スライド誤検知抑制ガード)」の正しい組み合わせで再走査する
# (2026-08-22 v2、user承認・00:25)。
#
# v1 (_rescan_zenchi_slide_exit_guard_2026-08-22.sh) からの変更点:
#   1. --kill-override-chain-completion を追加 (v1で抜けていた、coordinator
#      指摘の事故)。短区間確認 (t=4914.533/t=6717.5) で組み合わせ効果を
#      実証済み (t=4914.533は213→0件、t=6717.5は49→4件=87%減)。
#   2. 起動前にフラグをログへ出し、必須フラグの有無を grep で自動検証
#      してから起動する (「私が手書きで5つ落とした事故」再発防止、
#      memory feedback_use_single_source_for_flags_2026-08-22)。
#   3. 各区間完走後、scripts/_verify_zenchi_segment_completeness_2026-08-22.py
#      で実測終点が要求終点に到達しているかを自動検証し、不足区間だけ
#      自動再走査する (最大2回リトライ)。
#      根治: visualize_advantage_overlay.py:4893-4895 の cap.read() 失敗時
#      サイレントbreak (今回は触らない、既知の別課題として記録のみ)。
#
# 出力は data/verify/zenchi_render_slide_exit_guard_v2_2026-08-22/ (新規、
# v1/旧本番dumpは一切上書きしない)。
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
OUTDIR=data/verify/zenchi_render_slide_exit_guard_v2_2026-08-22
LOGDIR=logs/zenchi_rescan_slide_exit_guard_v2_2026-08-22
WARMUP=30
MAX_RETRIES=2

mkdir -p "$OUTDIR" "$LOGDIR"

BOUNDS=(0 893.7 1738.3 2637.3 3626.0 4379.5 5255.6 6131.6 7033.6)
N=$((${#BOUNDS[@]} - 1))

ADOPTED=$(PYTHONPATH=. ./venv/bin/python -c \
  "import src.production_config as pc; print(pc.advantage_overlay_flags())")
if [ -z "$ADOPTED" ]; then
  echo "=== 中止: advantage_overlay_flags() が空文字を返した (production_config.py 側の異常) ==="
  exit 1
fi

FLAGS="--layout panel --panel-subtitle-h 0 --no-force-in-match --no-render \
--model-dir $MODEL --warmup-sec $WARMUP \
--resolved-exchange-eval --resolved-decisive-amplify --resolved-live-defender \
--kill-override-chain-completion \
--enable-slide-exit-min-display-guard \
$ADOPTED"

echo "=== 30先動画 修正①+③組み合わせ 全編再走査(v2) start $(date +%F_%T) ==="
echo "[単一情報源] $ADOPTED"
echo "[構成] $FLAGS"
cat /proc/loadavg

# --- 起動前フラグ検証 (必須フラグが全部入っているか、累積フラグが混入していないか) ---
FLAG_CHECK_FAIL=0
for required in "--kill-override-chain-completion" "--enable-slide-exit-min-display-guard" \
                 "--resolved-exchange-eval" "--resolved-decisive-amplify" "--resolved-live-defender" \
                 "--layout panel" "--no-force-in-match" "--warmup-sec 30" \
                 "--production-recognition" "--resolved-kill-override" "--resolved-live-defender-strict"; do
  if echo "$FLAGS" | grep -qF -- "$required"; then
    echo "[OK] 必須フラグ確認: $required"
  else
    echo "[NG] 必須フラグ欠落: $required"
    FLAG_CHECK_FAIL=1
  fi
done
if echo "$FLAGS" | grep -qF -- "--kill-override-chain-gen-accumulate"; then
  echo "[NG] 累積フラグが混入している (付けない指示のはず)"
  FLAG_CHECK_FAIL=1
else
  echo "[OK] 累積フラグ(--kill-override-chain-gen-accumulate)は含まれていない"
fi
if [ "$FLAG_CHECK_FAIL" -ne 0 ]; then
  echo "=== 中止: フラグ検証NG。起動しない ==="
  exit 1
fi
echo "=== フラグ検証OK。起動します ==="

# --- 1区間分の実行 (関数化してリトライでも使い回す) ---
run_segment() {
  local i="$1" S="$2" E="$3"
  local out="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.mp4"
  local dump="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.npz"
  local log="$LOGDIR/seg$(printf '%02d' "$i").log"
  echo "[起動] 区間$i  $S 〜 $E 秒 -> $dump"
  (
    echo "=== seg$i start $(date +%F_%T) 範囲 $S〜$E 暖機 ${WARMUP}s ==="
    echo "[flags] $FLAGS"
    PYTHONPATH=. ./venv/bin/python -m scripts.visualize_advantage_overlay \
      --video "$VIDEO" --start-sec "$S" --end-sec "$E" \
      $FLAGS --dump-timeline "$dump" --out "$out"
    echo "=== seg$i done rc=$? $(date +%F_%T) ==="
  ) > "$log" 2>&1 &
}

# --- 完走検証 (不足していた区間の index リストを標準出力に1行ずつ返す) ---
verify_all_segments() {
  local incomplete=()
  for i in $(seq 1 "$N"); do
    local S=${BOUNDS[$((i - 1))]}
    local E=${BOUNDS[$i]}
    local dump="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.npz"
    if PYTHONPATH=. ./venv/bin/python -m scripts._verify_zenchi_segment_completeness_2026-08-22 \
        --npz "$dump" --expected-end-sec "$E"; then
      :
    else
      incomplete+=("$i")
    fi
  done
  echo "${incomplete[@]}"
}

# --- 1回目: 全8区間並列起動 ---
for i in $(seq 1 "$N"); do
  run_segment "$i" "${BOUNDS[$((i - 1))]}" "${BOUNDS[$i]}"
done
echo "--- 全${N}区間を起動、完了待ち ---"
wait
echo "=== 全区間完了 (1回目) $(date +%F_%T) ==="
grep -h "done rc=" "$LOGDIR"/seg*.log

# --- 完走検証+リトライループ ---
retry_count=0
while true; do
  echo "=== 完走検証 (試行 $retry_count) $(date +%F_%T) ==="
  incomplete_list=$(verify_all_segments)
  if [ -z "$incomplete_list" ]; then
    echo "=== 全区間 完走確認OK ==="
    break
  fi
  echo "[不足区間] $incomplete_list"
  if [ "$retry_count" -ge "$MAX_RETRIES" ]; then
    echo "=== 最大リトライ回数(${MAX_RETRIES})に到達。以下の区間が不足したまま ==="
    echo "$incomplete_list"
    break
  fi
  retry_count=$((retry_count + 1))
  echo "=== 不足区間を再走査 (${retry_count}/${MAX_RETRIES}回目) $(date +%F_%T) ==="
  for i in $incomplete_list; do
    run_segment "$i" "${BOUNDS[$((i - 1))]}" "${BOUNDS[$i]}"
  done
  wait
  echo "=== 再走査完了 $(date +%F_%T) ==="
done

# --- 最終集計 (row数は参考値。本フラグの根治で settled 再計算回数自体が
# 減るため旧基準142,930との厳密一致は理論上想定しない。完走判定は上の
# last t_sec ベースの verify_all_segments が本体) ---
echo "=== 最終集計 $(date +%F_%T) ==="
PYTHONPATH=. ./venv/bin/python -c "
import numpy as np, glob
files = sorted(glob.glob('$OUTDIR/seg*.npz'))
total = 0
for f in files:
    d = np.load(f, allow_pickle=True)
    n = len(d['t_sec'])
    total += n
    print(f, n)
print('TOTAL_ROWS', total)
print('参考: 旧基準(112エピソード版)は142930行だが、本修正でsettled再計算')
print('回数自体が変わるため厳密一致は想定しない。完走判定は上記verify参照。')
"
ls -l "$OUTDIR"
echo "--- 各区間の終了コード (最終ログ) ---"
grep -h "done rc=" "$LOGDIR"/seg*.log
echo "=== 全処理完了 $(date +%F_%T) ==="
