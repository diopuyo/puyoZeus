#!/bin/bash
# T2 の完了を待ち、npz 全列一致なら 48本収集を自動で開始する (2026-08-20)。
#
# 手順:
#   1. T2 収集 (39番の ON/OFF) の完了を待つ
#   2. npz 全列が bit 一致するか検証
#   3. 合格 → native フラグを production_config に登録して 48本収集を起動
#      不合格 → **収集は起動せず停止**して報告 (既定 OFF のままなので安全)
#
# 不合格時に「フラグ無しで収集を始める」自動フォールバックは**入れない**。
# 不一致が出たなら原因を調べるのが先で、黙って別構成の収集を 15 時間走らせる
# のは事故 (今日の配線漏れ事故と同じ構図になる)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
exec >> logs/t2_then_collect_2026-08-20.log 2>&1
echo "=== 待機開始 $(date +%F_%T) ==="

# 1. T2 の完了待ち
while pgrep -f "boards_lean_t2_" > /dev/null; do
  sleep 60
done
echo "--- T2 収集完了 $(date +%T) ---"
cat logs/t2_native_parity_2026-08-20.log | tail -4

# 2. 全列一致の検証
echo "--- npz 比較 $(date +%T) ---"
PYTHONPATH=. ./venv/bin/python -m scripts._verify_t2_npz_identical_2026-08-20 39
RC=$?
echo "  検証 rc=$RC"

if [ "$RC" -ne 0 ]; then
  echo "=== T2 不合格: 48本収集は起動しない。原因調査が先 ==="
  echo "    (native は既定 OFF のままなので本番構成は無変更)"
  exit 1
fi

# 3. 採用登録は main Claude が行う (根拠の記録が必要なため自動化しない)。
#    ここではフラグ登録済みかを確認してから起動する。
echo "--- 採用登録の確認 $(date +%T) ---"
if ! grep -q "enable-native-hsv-classifier" src/production_config.py; then
  echo "=== production_config に未登録。収集は起動しない ==="
  echo "    T2 は合格しているので、登録してから本スクリプトを再実行すること"
  exit 2
fi
CF=$(PYTHONPATH=. ./venv/bin/python -c "from src.production_config import collect_flags; print(collect_flags())")
case "$CF" in
  *--enable-native-hsv-classifier*) echo "  collect_flags に含まれている" ;;
  *) echo "=== collect_flags に載っていない (登録が COLLECT_ONLY 以外?)。起動しない ==="; exit 3 ;;
esac

# 4. 48本収集を起動
echo "--- 48本収集 起動 $(date +%T) ---"
rm -f data/verify/regen_2026-08-20_model50v2/status.tsv
setsid -f bash scripts/_run_model50v2_2026-08-20.sh < /dev/null
sleep 40
echo -n "  収集プロセス: "
pgrep -c -f _collect_lean_1t
echo "  native フラグが渡っているか:"
pgrep -af _collect_lean_1t | head -1 | tr ' ' '\n' | grep -c "enable-native-hsv-classifier"
echo "=== 完了 $(date +%F_%T) ==="
