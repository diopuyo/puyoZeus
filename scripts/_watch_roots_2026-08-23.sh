#!/bin/bash
# 根治作業の定期報告 (2026-08-23)。
# 変数の展開タイミングを誤らないよう、条件はスクリプト内で毎回評価する
# (本日、$(...) が仕掛けた時点で展開されて待機ループが壊れた事故があった)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
sleep 1500
echo "=== 定期報告 $(date +%H:%M) ==="
cat /proc/loadavg
echo "--- 変更中のファイル ---"
git status --short | grep -E "scan_judgment|visualize_adv|_diag_" | head -5
echo "--- 新しいログ ---"
ls -t logs/ 2>/dev/null | head -4
