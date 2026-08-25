#!/bin/bash
# 30先2セット動画 (117分) の本番レンダ: 試合開始で8区間に分けて並列実行 (2026-08-21)。
#
# ## 分割点の根拠
#
# 試合開始で切る = 試合開始のときに状態機械がリセットされるので前の状態を
# 引き継ぐ必要がない (user 指示 2026-08-21)。候補7点は目視で ±0秒 の精度を確認済み。
#
# **区間4がセット境界 (t=3626s) でちょうど終わる**ので、
#   セット1 = 区間1〜4 / セット2 = 区間5〜8
# を結合するだけで納品用の2本になる (再エンコード不要)。
#
# セット境界の根拠 (目視確認済み):
#   t≈3483s で WIN星 29→30 (のらすけ選手が30本先取、13連勝、通算02201950)
#   t≈3490〜3625s に約140秒のリザルト+キャラ再選択+ロード演出
#   t≈3626s でスコア0にリセット→「レディ」、2P名が「のらすけしんえいたい」に変化
#
# ## 暖機 30秒 の根拠
#
# 実測 (scripts/_cmp_warmup_cause_2026-08-21.py) で、通しレンダとの一致は:
#   暖機 0秒: 判定の色が変わる行が 48件 (16.1%) / 盤面のずれ 9件
#   暖機 5秒: 8件 (2.7%) / 盤面のずれ 6件
#   暖機26秒: **0件 (0.00%) / 盤面のずれなし**
# 26秒で収束するので余裕を見て30秒。残る微差は平滑化(EMA)の内部状態のみで
# 互角±3.0 を一度も跨がない = 見た目に出ない。
#
# ## 並列数 8 の根拠
#
# 実測スループット (動画秒/壁秒、フル書き出し・実ディスクI/O込み):
#   P=2: 0.2294 / P=4: 0.4410 / P=6: 0.4613 / P=8: 0.6680
# 8まで一貫して増加 (効率は落ちるが総スループットは最良)。
#
# ## フラグ構成 (2026-08-22 是正: 単一情報源化)
#
# **配線事故 (7件目、2026-08-22 user指摘)**: 従来この FLAGS は手書きだった。
# CLAUDE.md「採用フラグは src/production_config.py が単一情報源」に反しており、
# 実際に本番採用済み5フラグ (--per-side-settled/--no-score-lead-bias/
# --no-pressure/--sample-interval 0/--resolved-kill-override) が抜け落ちて
# いた (production_config.advantage_overlay_flags() の出力と突合して発覚)。
# 今後同じ事故を防ぐため、$ADOPTED を production_config.advantage_overlay_
# flags() から実行時に取得し、そのまま FLAGS に埋め込む形に是正する
# (--early-fire-reaction もここに含まれるため個別の付け忘れ注意書きは不要に
# なった)。
#
# --show-recognition は付けない (user 指定: 中央は元映像、認識色の重畳なし)。
# --exclude-video は絶対に付けない (付けると --model-dir が無視され CSV 起動時学習に落ちる)。
#
# ## 親3フラグ (--resolved-exchange-eval/--resolved-decisive-amplify/
#    --resolved-live-defender) の手書きは 2026-08-24 に削除済み
#
# この3つは 2026-08-13〜15 に実装・全デモ/レビュー動画で継続使用されていたが
# production_config.py には子フラグ (--resolved-live-defender-strict/
# --resolved-kill-override) だけが登録され親3つが未登録という不整合があり、
# 本スクリプトが手書きで補っていた (手書き自体が配線事故7件目と同型の温床)。
# 2026-08-24 の user 承認で親3つが ADVANTAGE_ADOPTED へ正式登録され $ADOPTED に
# 含まれるようになったため、手書き行を削除した (重複指定でも argparse は
# 壊れないが、単一情報源の原則に反するため残さない)。
#
# ## --no-force-in-match を $ADOPTED に含めない理由
#
# ADVANTAGE_ADOPTED は「常に有効にすべき品質改善」のみを登録する場所であり、
# --no-force-in-match は「試合境界を跨ぐフル動画レンダ専用」のモード選択肢
# (generate() docstring :3972、既定 True=通常単発試合レンダ向け) であって
# 品質改善ではない。本スクリプトのような複数試合を跨ぐレンダでのみ必要な
# ローカル設定として今後も手書きのままにする (同種の他フラグは調査済み、
# argparse 全項目中この意味論を持つのは本フラグのみ)。
#
# ## --sample-interval 0 の再レンダ所要時間への影響 (2026-08-22 実測確認)
#
# sample_interval は writer.write() (実フレーム書き出し) を一切間引かない
# (毎フレーム書き出しは既に無条件、:4519 の step は history グラフの記録
# 頻度のみを制御)。ただし `_draw_graph` は history 全点を毎フレーム
# Python ループで描画するため、sample-interval 0 (=間引きなし、記録点が
# 約4〜5倍に増える) では試合が長いほど1試合内で二次関数的にグラフ描画コスト
# が増える可能性がある (history は試合境界でクリアされるため無制限には
# 増えない)。本番採用の理由 (「0.5秒間引きだとおじゃま会計が取りこぼす」)
# 自体は現在の実装 (_drive_ojama が sample_interval非依存で毎フレーム駆動)
# では該当しなくなっているが、**フラグ自体は登録済み採用構成のため外さず
# 使う**。短区間実測は scripts/_verify_sample_interval_cost_2026-08-22.py
# 参照 (フルレンダ規模への影響は自己検証で確認、詳細は引き継ぎ参照)。
#
# ## スレッド制限 (2026-08-21 23:53 是正、投入1回目の失敗を受けて)
#
# 1回目の投入では各プロセスが **109スレッド** を立て、8プロセスで872スレッドが
# 16コアを取り合った。負荷59、実効スループット 0.1166 動画秒/壁秒 =
# 事前実測 (0.668) の **5.7分の1**、完了見込み16.5時間で使用不能だった。
#
# 原因: `scripts/visualize_advantage_overlay.py` にも `src/` の認識経路にも
# スレッド制限が入っていない。収集側は「cv2.setNumThreads(1)×14並列が最適」と
# 実測済み (memory project_collect_indicators_v2_perf) だが、レンダ経路は未適用だった。
#
# 対処: 環境変数で BLAS 系と OpenCV のスレッドを1に固定する。
# 1プロセス1コアにして、並列数でスケールさせる思想 (収集と同じ)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1

# BLAS/OpenMP 系のスレッドを1に固定 (過剰生成の防止)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# OpenCV のスレッドも1に (ビルドによって効く変数が違うので両方指定)
export OPENCV_FOR_THREADS_NUM=1
export CV_NUM_THREADS=1

VIDEO=data/frames/video_zenchi_c0BQoMJwwQU.mp4
MODEL=data/verify/retrain_model62_2026-08-21
OUTDIR=data/verify/zenchi_render_2026-08-21
LOGDIR=logs/zenchi_render_2026-08-21
WARMUP=30

mkdir -p "$OUTDIR" "$LOGDIR"

# 区間の境界 (秒)。試合開始時刻。0 と 7033.6 は動画の両端。
# 3626.0 がセット境界 (区間4の終わり = セット1の終わり)。
BOUNDS=(0 893.7 1738.3 2637.3 3626.0 4379.5 5255.6 6131.6 7033.6)
N=$((${#BOUNDS[@]} - 1))

# 単一情報源から採用済みフラグ一式を取得する (2026-08-22 是正、CLAUDE.md
# 「採用フラグは src/production_config.py が単一情報源」)。手書き FLAGS への
# 個別追記はここで終わり、以後の採用フラグ追加は production_config.py の
# ADVANTAGE_ADOPTED を更新するだけで本レンダにも自動反映される。
ADOPTED=$(PYTHONPATH=. ./venv/bin/python -c \
  "import src.production_config as pc; print(pc.advantage_overlay_flags())")
if [ -z "$ADOPTED" ]; then
  echo "=== 中止: advantage_overlay_flags() が空文字を返した (production_config.py 側の異常) ==="
  exit 1
fi

# 2026-08-23 追加: 判定の誤り (全編112エピソード・395.3秒 = 動画の5.6%) の修正2件。
#
# --kill-override-chain-completion:
#   致死の安全弁が「連鎖前の凍結盤面の空き」と「額面の予告」で判定していたのを、
#   連鎖完走後の空きと相殺後の残量で判定する。t=6717.5 (1Pの15連鎖) で
#   空きが 5→62 に是正され、予告216が相殺されて余剰594を相手へ反撃に回る。
#
# --enable-slide-exit-min-display-guard:
#   真因。ネクストの「スライド動き」検知が連鎖の演出で1.37秒周期に誤検知し、
#   保持時間の式を迂回して連鎖イベントを強制終了 → 断片化 → 連鎖数誤検知(15→5)
#   → 火力10分の1 (84 vs 840) という鎖の起点だった。
#   色分類 (精度100%) との突合で誤検知だけを弾く。信号自体は殺さない
#   (user 伝授の絶対律「連鎖の終わり = 連鎖側のネクストが動いた瞬間」を守る)。
#
# 効果 (全編再走査、8/23): 合計時間 395.3秒 → 188.8秒 (-52%)、解消49件。
# 認識精度は OFF/ON で 1ビットも変わらない (全マス合意率0.9971、不一致5,896)。
# 「新規32件」は比較ツールが是正後の値を見ていないための境界アーティファクトで、
# 実画面5枚+直接計装で「真に新規で本物は0件」を確認済み。
FLAGS="--layout panel --panel-subtitle-h 0 --no-force-in-match \
--model-dir $MODEL --warmup-sec $WARMUP \
--kill-override-chain-completion --enable-slide-exit-min-display-guard \
$ADOPTED"

echo "=== 30先動画 本番レンダ start $(date +%F_%T) ==="
echo "[単一情報源] $ADOPTED"
echo "[構成] $FLAGS"
echo "[区間] $N 区間、並列 $N、暖機 ${WARMUP}秒"
echo "[分割] 区間1-4 = セット1 (0〜3626.0s) / 区間5-8 = セット2 (3626.0〜7033.6s)"
cat /proc/loadavg

for i in $(seq 1 "$N"); do
  S=${BOUNDS[$((i - 1))]}
  E=${BOUNDS[$i]}
  OUT="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.mp4"
  DUMP="$OUTDIR/seg$(printf '%02d' "$i")_${S}_${E}.npz"
  LOG="$LOGDIR/seg$(printf '%02d' "$i").log"
  echo "[起動] 区間$i  $S 〜 $E 秒 (長さ $(echo "$E - $S" | bc)s) -> $OUT"
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
