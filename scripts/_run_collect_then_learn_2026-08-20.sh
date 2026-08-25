#!/bin/bash
# 追加14本の収集完了を待って学習工程まで自動で進める (2026-08-21 更新)。
#
# 安全弁:
#   - 62本揃っていなければ**学習を起動しない** (欠測のまま学習すると前回結果と
#     比較できなくなる、手順書 docs/RECOLLECT148_2026-08-18.md の警告)
#   - 品質ゲートの結果はログに残すが、FAIL 動画の除外は**しない**
#     (除外するか目視に回すかは user 判断、と手順書に明記されている)
#   - --exclude-match-end-locked は _run_model50v2_learning に埋め込み済み
#     (付け忘れても警告されない仕様のため)
#
# 待機条件の注意 (2026-08-21 の事故を踏まえて):
#   `pgrep -f "boards_lean_model50v2"` のような**広いパターンは使わない**。
#   npz ディレクトリ名は診断コマンドや他の監視ジョブのコマンドラインにも現れる
#   ため、それらを収集プロセスと誤検出して待機ループから抜けられなくなった
#   (実際に 48本収集で 55分ロスした)。オーケストレータのモジュール名だけを見る。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/collect_then_learn_2026-08-21.log 2>&1
echo "=== 待機開始 $(date +%F_%T) ==="

NPZ_DIR=data/indicators_v2/boards_lean_model50v2_2026-08-20
TARGET=62

while pgrep -f "_regen_add14_2026-08-21" > /dev/null; do
  N=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] 収集中... 完了 ${N}/${TARGET}"
  sleep 300
done

N=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | wc -l)
echo "--- 収集終了 $(date +%F_%T) 完了 ${N}/${TARGET} ---"
tail -5 logs/regen_add14_2026-08-21.log

if [ "$N" -lt "$TARGET" ]; then
  echo "=== ${TARGET}本に届いていない (${N}本)。学習は起動しない ==="
  echo "    欠測のまま学習すると動画数が減った分だけ前回と比較できなくなる"
  echo "    (docs/RECOLLECT148_2026-08-18.md の警告)。原因を確認すること。"
  exit 1
fi

echo "--- 学習工程 起動 $(date +%T) ---"
bash scripts/_run_model50v2_learning_2026-08-20.sh
echo "=== 完了 $(date +%F_%T) ==="
