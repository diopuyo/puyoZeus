#!/bin/bash
# 結合とh264の完了を待って状態を出す (2026-08-23)。
# 変数の展開タイミングを誤らないよう、条件はスクリプト内で毎回評価する。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
sleep 1500
echo "=== 定期報告 $(date +%H:%M) ==="
cat /proc/loadavg
echo "--- 結合 ---"
tail -3 logs/zenchi_concat_2026-08-23.log 2>/dev/null
echo "--- h264 ---"
tail -3 logs/zenchi_h264_2026-08-23.log 2>/dev/null
echo "--- 納品物 ---"
ls -l --time-style=+%H:%M data/verify/zenchi_delivery_2026-08-21/*h264*.mp4 2>/dev/null
