#!/bin/bash
# 不慮の Claude 切断後、 1 コマンドで現状把握。
# 使い方: wsl -d Ubuntu -- bash scripts/_status_check.sh
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

echo "================================================================"
echo "  自律実行 status check @ $(date)"
echo "================================================================"
echo ""
echo "## 現在走行中のプロセス"
pgrep -af python | grep -E "extract_hsv|visualize_recognition|evaluate_recognition|autonomous|phase_i_fine" || echo "  (none)"
echo ""
echo "## 完了 flag"
for f in logs/baseline_v3_eval/all_done.flag \
         logs/cycle50_seed_regen/all_done.flag \
         logs/cycle51_ojama_dryrun/all_done.flag \
         logs/autonomous_sweep_main/all_done.flag \
         logs/autonomous_master/all_done.flag; do
  if [ -f "$f" ]; then echo "  ✓ $f"
  else echo "  ✗ $f"
  fi
done
echo ""
echo "## ダッシュボード"
if [ -f logs/autonomous_dashboard.md ]; then
  head -60 logs/autonomous_dashboard.md
else
  echo "  (まだ dashboard 未生成)"
fi
echo ""
echo "## 直近 judgments"
if [ -f data/verify/autonomous/_judgments.jsonl ]; then
  wc -l data/verify/autonomous/_judgments.jsonl
  tail -5 data/verify/autonomous/_judgments.jsonl
else
  echo "  (まだ judgments なし)"
fi
echo ""
echo "## 自律 master 起動の仕方 (= 落ちた場合)"
echo "  setsid -f bash scripts/_autonomous_master.sh > logs/autonomous_master.log 2>&1 < /dev/null"
echo ""
