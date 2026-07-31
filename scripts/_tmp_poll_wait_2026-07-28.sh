#!/bin/bash
# #43 段階1評価のバックグラウンドジョブ完了待ちポーリング。
# MSYS 経由 wsl でパイプ文字("|")が壊れる問題 (feedback_msys_pipe_escape.md) を
# 回避するため、複数パターンを個別 pgrep -c -f で判定する (正規表現 OR を使わない)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
while true; do
  n1=$(pgrep -c -f relphase_win_auc_generic)
  n2=$(pgrep -c -f _tmp_video_phase_auc_breakdown)
  total=$((n1 + n2))
  if [ "$total" -eq 0 ]; then
    echo "ALL_DONE"
    break
  fi
  sleep 5
done
