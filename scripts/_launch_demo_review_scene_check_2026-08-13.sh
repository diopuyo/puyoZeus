#!/bin/bash
# _demo_review_scene_check_2026-08-13.sh を setsid detach して起動するだけの
# 薄いランチャ (git-bash -> wsl のネスト引用符崩れ回避、MSYSパイプ注意事項準拠)。
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
setsid -f bash scripts/_demo_review_scene_check_2026-08-13.sh \
  > logs/demo_review_scene_check_all_2026-08-13.log 2>&1 < /dev/null &
disown
echo "LAUNCHED pid=$!"
