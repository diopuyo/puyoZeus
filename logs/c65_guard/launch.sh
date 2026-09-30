#!/bin/bash
# 影子測定の起動 (3レーン並列)
cd /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
L=logs/c65_guard/lane.sh
setsid -f bash $L q_7gc4TgFig zenchi < /dev/null > /dev/null 2>&1
setsid -f bash $L fcXG83vInDY < /dev/null > /dev/null 2>&1
setsid -f bash $L mia8KCjr52g < /dev/null > /dev/null 2>&1
