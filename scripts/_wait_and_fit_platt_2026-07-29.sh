#!/bin/bash
# extra4 収集(14ジョブ)の完走を待ってから Platt 校正器を学習する。
#
# 経緯: 2026-07-29 16時台に fit_platt_calibration を起動したが、エージェントの
# バックグラウンドジョブとして起動されており setsid detach していなかったため
# 親の終了とともに kill され、校正器 JSON が生成されなかった。
# 本スクリプトは CLAUDE.md のプロセス管理ルールに従い setsid detach して起動する。
#
# 収集完走を待つ理由: 学習は 73,416 行 x 132 特徴量の GroupKFold で、収集14並列と
# 競合すると 45 分以上かかって終わらなかった。CPU が空いてから回す。
set -e
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
export PYTHONPATH=.

LOG=logs/fit_platt_calibration_2026-07-29.log
OUT=data/indicators_v2/platt_calibration.json

echo "[wait-platt] $(date) extra4収集の完走待機開始 (PID=$$)" | tee -a "$LOG"

# 収集プロセスが消えるまで待つ
while pgrep -f "_collect_1t" > /dev/null; do
  sleep 60
done
echo "[wait-platt] $(date) 収集プロセスの消失を検知" | tee -a "$LOG"

# 念のため CPU が落ち着くまで待つ
sleep 60

echo "[wait-platt] $(date) 校正器の学習を開始" | tee -a "$LOG"
nice -n 10 ./venv/bin/python -m scripts.fit_platt_calibration >> "$LOG" 2>&1
rc=$?
echo "[wait-platt] $(date) 学習終了 (exit=$rc)" | tee -a "$LOG"

# fail-silent 防止: 出力が無ければエラーを明示して異常終了する
if [ ! -f "$OUT" ]; then
  echo "[ERROR] 校正器 JSON が生成されていない: $OUT" | tee -a "$LOG"
  echo "[wait-platt] ERROR DONE" | tee -a "$LOG"
  exit 1
fi
echo "[wait-platt] ALL DONE ($OUT)" | tee -a "$LOG"
