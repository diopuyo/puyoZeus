#!/bin/bash
# 自分の並走を止めた単独計測 (b20 の 5830〜7030 秒): exact → B2 → B1 の順に 1 プロセスずつ
S=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed
R=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
VARIANT=exact bash $S/run_variant.sh $R clean_exact live b20_on
VARIANT=bounded2 bash $S/run_variant.sh $R clean_bounded2 live b20_on
VARIANT=bounded bash $S/run_variant.sh $R clean_bounded live b20_on
