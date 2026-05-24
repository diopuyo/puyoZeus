#!/bin/bash
# cycle_10 (0.70) 完走後に cycle_11 (0.80) を自動起動する
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer

# cycle_10 完走待ち (5 動画 mp4 全部できるまで)
echo "[chain] waiting cycle_10 completion..."
while [ $(ls data/test_unknown/*_viz_multicycle_10.mp4 2>/dev/null | wc -l) -lt 5 ]; do
  # cycle_10 process が全消滅したら break (異常終了対応)
  if ! pgrep -f "multi_video_cycle.*cycle 10\|visualize_recognition.*multicycle_10" > /dev/null 2>&1; then
    sleep 10
    # 念のため再確認
    if ! pgrep -f "multi_video_cycle.*cycle 10\|visualize_recognition.*multicycle_10" > /dev/null 2>&1; then
      echo "[chain] cycle_10 procs gone, 5 mp4 not present yet — assume crash"
      break
    fi
  fi
  sleep 30
done
echo "[chain] cycle_10 done at $(date)"

# cycle_11 起動
echo "[chain] launching cycle_11..."
PYTHONPATH=. ./venv/bin/python -m scripts.multi_video_cycle --cycle 11 --parallel 3 --cnn-override-prob 0.80 > logs/multi_video_cycle_11.log 2>&1
echo "[chain] cycle_11 done at $(date)"
