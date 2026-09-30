#!/bin/bash
# 有界化の速度確認 (b20 の 5830〜7030 秒): B2 → B1
S=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed
R=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
VARIANT=bounded2 bash $S/run_variant.sh $R live_bounded2 live b20_on
VARIANT=bounded bash $S/run_variant.sh $R live_bounded live b20_on
