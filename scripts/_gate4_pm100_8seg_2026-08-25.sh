#!/bin/bash
# Gate 4 "pre-gate measurement": PM100 (30先2セット zenchi 動画) 全8区間、
# 条件1〜4 の timeline dump 取得ハーネス。
#
# 【表記上の注意】本スクリプトが生成するのは Gate 4 の正式検収データではなく
# "pre-gate measurement" (Codex 許可 2026-08-25、docs/agent_coordination/
# CLAUDE_TO_CODEX.md 第25報【追記2】)。合否判定・本番ON・production 登録は
# 一切行わない。Gate 3R-6 PASS 後にあらためて正式な Gate 4 検収を行う。
#
# 2026-08-25 更新点 (Codex 指定の隔離条件対応):
#   - 別エージェントが編集中の src/death_confirmation.py /
#     tests/test_death_confirmation.py / scripts/visualize_advantage_overlay.py
#     を含む可変な作業ツリーを直接の測定入力にしない。固定 snapshot
#     ($SNAP、scripts/_gate4_pregate_make_snapshot_2026-08-25.sh で作成済み)
#     だけを読む。
#   - フラグ構成は単一情報源からの動的合成をやめ、Codex が原本スクリプトを
#     復元した BASEFLAGS をそのまま固定で使う (推測・再導出しない)。
#   - 出力先・ログ先を新しい日付ディレクトリ (gate4_pregate_pm100_2026-08-25)
#     に分離し、旧 gate4_pm100_2026-08-25/ には一切書かない。
#   - PYTHONHASHSEED=0 を追加。
#   - 並列度の上限を 3 に強制 (Codex 指定)。
#
# ## 条件1〜4 の定義 (Codex 復元、そのまま使用)
#
# | 条件 | 内容                     | 追加フラグ                                        |
# |------|--------------------------|----------------------------------------------------|
# | 1    | 旧OFF基準                | (なし)                                              |
# | 2    | 現行ヒステリシスBのみ    | --kill-override-hysteresis                          |
# | 3    | 現行規模比較Aのみ        | --kill-override-scale-compare                       |
# | 4    | 現行A+B                  | --kill-override-hysteresis --kill-override-scale-compare |
# | 5    | 交換台帳等 (未着手)      | 対象外。関数内で exit 1 する (Gate 3R-6 PASSまで禁止) |
#
# ## 使い方 (単独区間の試走。所要時間の実測用):
#
#   MSYS_NO_PATHCONV=1 wsl -d Ubuntu -- bash -c \
#     'cd /mnt/c/.../puyo_analyzer && \
#      bash scripts/_gate4_pm100_8seg_2026-08-25.sh 1 1 1 1'
#     (引数: 条件番号(1-4) 開始区間 終了区間 並列度(<=3))
#
#   # 条件1を全8区間、逐次で実行:
#   bash scripts/_gate4_pm100_8seg_2026-08-25.sh 1 1 8 1
#
#   # 全4条件を setsid detach で逐次投入 (WSL再起動不要、長時間放置前提):
#   wsl -d Ubuntu -- bash -c \
#     "cd ... && setsid -f bash scripts/_run_gate4_pm100_all_2026-08-25.sh \
#      > logs/gate4_pregate_pm100_2026-08-25/all.log 2>&1 < /dev/null"
#
set -u
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

COND="${1:-}"
SEG_START="${2:-1}"
SEG_END="${3:-8}"
PARALLEL="${4:-1}"

if [ -z "$COND" ]; then
  echo "使い方: $0 <条件番号 1-5> [開始区間 既定1] [終了区間 既定8] [並列度 既定1・上限3]"
  exit 1
fi

# 並列度の上限を 3 に強制 (Codex 指定、判断依頼1の許可条件)。
if [ "$PARALLEL" -gt 3 ] 2>/dev/null; then
  echo "=== 中止: 並列度 $PARALLEL は上限3を超える (Codex指定) ==="
  exit 1
fi

# BLAS/OpenMP/OpenCV のスレッドを1に固定 + PYTHONHASHSEED 固定
# (Codex 指定の manifest 項目、再現性確保)。
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1
export CV_NUM_THREADS=1

# 固定 snapshot (可変な作業ツリーを直接の測定入力にしない、Codex 指定)。
# scripts/_gate4_pregate_make_snapshot_2026-08-25.sh で作成済み。存在しない
# 場合は実行しない (勝手に作り直さない、事前に snapshot 作成手順を踏むこと)。
SNAP=data/verify/gate4_pregate_pm100_2026-08-25/_snapshot_20260825_2300jst
if [ ! -f "$SNAP/scripts/visualize_advantage_overlay.py" ]; then
  echo "=== 中止: snapshot が見つからない ($SNAP) ==="
  echo "先に scripts/_gate4_pregate_make_snapshot_2026-08-25.sh を実行すること"
  exit 1
fi

VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
MODEL=data/verify/retrain_model62_2026-08-21
WARMUP=30

# 区間境界 (旧スクリプト・pm100_fix_2026-08-24 と完全同一。before/after を
# 同じ切り口で比較するため変更しない)。
BOUNDS=(0 893.7 1738.3 2637.3 3626.0 4379.5 5255.6 6131.6 7033.6)

OUTBASE=data/verify/gate4_pregate_pm100_2026-08-25
LOGBASE=logs/gate4_pregate_pm100_2026-08-25
mkdir -p "$LOGBASE"

# Codex 復元の BASEFLAGS (docs/agent_coordination/CLAUDE_TO_CODEX.md 第25報
# 【追記2】判断依頼2で提示、単一情報源からの動的合成をやめ手書き禁止の
# 例外としてそのまま固定する。2026-08-25 実測で
# src.production_config.advantage_overlay_flags() の出力と BASEFLAGS の
# early-fire-reaction 以降19個が完全一致することを確認済み、manifest.txt
# 参照)。
BASEFLAGS="--layout panel --panel-subtitle-h 0 --no-force-in-match \
--model-dir $MODEL --warmup-sec $WARMUP \
--kill-override-chain-completion --enable-slide-exit-min-display-guard \
--early-fire-reaction --per-side-settled --no-score-lead-bias --no-pressure \
--sample-interval 0 --counter-reach --normalize-fps-30 \
--production-recognition --resize-1080p --resolved-exchange-eval \
--resolved-decisive-amplify --resolved-live-defender \
--resolved-live-defender-strict --resolved-kill-override"

case "$COND" in
  1) COND_NAME="cond1_off_baseline"; EXTRA="" ;;
  2) COND_NAME="cond2_hysteresis_only"; EXTRA="--kill-override-hysteresis" ;;
  3) COND_NAME="cond3_scale_compare_only"; EXTRA="--kill-override-scale-compare" ;;
  4) COND_NAME="cond4_a_plus_b"; EXTRA="--kill-override-hysteresis --kill-override-scale-compare" ;;
  5)
    echo "=== 条件5 (交換台帳+未解決ゲート+掛け算式根治) は Gate 3R-6 PASS まで実行禁止 ==="
    echo "(docs/agent_coordination/CLAUDE_TO_CODEX.md 第25報【追記2】判断依頼1)。中止する。"
    exit 1
    ;;
  *)
    echo "不明な条件番号: $COND (1-5 を指定)"
    exit 1
    ;;
esac

OUTDIR="$OUTBASE/$COND_NAME"
LOGDIR="$LOGBASE/$COND_NAME"
mkdir -p "$OUTDIR" "$LOGDIR"

FLAGS="--no-render $BASEFLAGS $EXTRA"

# 監査用の条件別補助ログ (全項目の正本は data/verify/gate4_pregate_pm100_2026-08-25/manifest.txt)。
{
  echo "生成日時: $(date +%F_%T)"
  echo "表記: pre-gate measurement (Gate 4 の正式検収ではない)"
  echo "snapshot: $SNAP"
  echo "条件: $COND ($COND_NAME)"
  echo "BASEFLAGS (Codex復元、固定): $BASEFLAGS"
  echo "EXTRA (A/Bトグル): $EXTRA"
  echo "全FLAGS: $FLAGS"
  echo "区間範囲: $SEG_START-$SEG_END / 並列度: $PARALLEL (上限3)"
  echo "PYTHONHASHSEED=$PYTHONHASHSEED"
} > "$LOGDIR/run_manifest.txt"

echo "=== [pre-gate measurement] Gate4 PM100 条件$COND ($COND_NAME) start $(date +%F_%T) ==="
echo "[snapshot] $SNAP"
echo "[構成] $FLAGS"
echo "[区間] $SEG_START〜$SEG_END / 並列度 $PARALLEL / 暖機 ${WARMUP}秒 (--no-render)"
cat /proc/loadavg

# 並列度 PARALLEL を守りつつ区間を逐次/低並列で処理する。
running=0
for i in $(seq "$SEG_START" "$SEG_END"); do
  S=${BOUNDS[$((i - 1))]}
  E=${BOUNDS[$i]}
  DUMP="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.npz"
  OUT="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.mp4"  # --no-render のため実際には作られない
  LOG="$LOGDIR/seg$(printf '%02d' "$i").log"
  echo "[起動] 区間$i  $S 〜 $E 秒 -> $DUMP"
  (
    T0=$(date +%s)
    echo "=== [pre-gate measurement] seg$i start $(date +%F_%T) 範囲 $S〜$E 暖機 ${WARMUP}s ==="
    echo "[flags] $FLAGS"
    # 直接ファイルパス実行 (-m scripts.xxx にしない): -m だと sys.path[0] が
    # cwd (リポジトリ直下) になり、可変な作業ツリー側の scripts/src が
    # snapshot より先に解決されてしまう (2026-08-25 気付き)。直接パス実行なら
    # インタプリタが sys.path[0]=$SNAP/scripts にし、スクリプト自身の
    # sys.path.insert(0, 親の親) で $SNAP がさらに先頭に入るため、
    # cwd や PYTHONPATH の値に関係なく snapshot だけが解決される。
    nice -n 19 ./venv/bin/python "$SNAP/scripts/visualize_advantage_overlay.py" \
      --video "$VIDEO" --start-sec "$S" --end-sec "$E" \
      $FLAGS --dump-timeline "$DUMP" --out "$OUT"
    RC=$?
    T1=$(date +%s)
    echo "=== seg$i done rc=$RC $(date +%F_%T) elapsed_sec=$((T1 - T0)) ==="
  ) > "$LOG" 2>&1 &
  running=$((running + 1))
  if [ "$running" -ge "$PARALLEL" ]; then
    wait -n 2>/dev/null || wait
    running=$((running - 1))
  fi
done
wait
echo "=== [pre-gate measurement] 条件$COND 完了 $(date +%F_%T) ==="
ls -l "$OUTDIR"
echo "--- 各区間の終了コード・所要時間 ---"
grep -h "done rc=" "$LOGDIR"/seg*.log
