#!/bin/bash
# 48本収集の完了を待って学習工程まで自動で進める (2026-08-20)。
#
# 収集完了は翌 10:15 頃の見込みで、そこから CSV ビルド 1.5時間 + 学習 32分 が
# 続く。人が待つ必要がない工程なので繋げておく。
#
# 安全弁:
#   - 48本揃っていなければ**学習を started しない** (欠測のまま学習すると
#     前回結果と比較できなくなる、手順書 docs/RECOLLECT148_2026-08-18.md の警告)
#   - 品質ゲートの結果はログに残すが、FAIL 動画の除外は**しない**
#     (除外するか目視に回すかは user 判断、と手順書に明記されている)
#   - --exclude-match-end-locked は _run_model50v2_learning に埋め込み済み
#     (付け忘れても警告されない仕様のため)
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/collect_then_learn_2026-08-20.log 2>&1
echo "=== 待機開始 $(date +%F_%T) ==="

NPZ_DIR=data/indicators_v2/boards_lean_model50v2_2026-08-20

# 収集プロセスが立ち上がるまで少し待つ (起動直後に呼ばれた場合の空振り防止)
sleep 120

while pgrep -f "_regen_model50v2_2026-08-20" > /dev/null || \
      pgrep -f "boards_lean_model50v2" > /dev/null; do
  N=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | wc -l)
  echo "[$(date +%H:%M)] 収集中... 完了 ${N}/48"
  sleep 600
done

N=$(ls "$NPZ_DIR"/*.npz 2>/dev/null | wc -l)
echo "--- 収集終了 $(date +%F_%T) 完了 ${N}/48 ---"
tail -5 logs/regen_model50v2_2026-08-20.log

if [ "$N" -lt 48 ]; then
  echo "=== 48本に届いていない (${N}本)。学習は起動しない ==="
  echo "    欠測のまま学習すると動画数が減った分だけ前回と比較できなくなる"
  echo "    (docs/RECOLLECT148_2026-08-18.md の警告)。原因を確認すること。"
  exit 1
fi

echo "--- 学習工程 起動 $(date +%T) ---"
bash scripts/_run_model50v2_learning_2026-08-20.sh
echo "=== 完了 $(date +%F_%T) ==="
