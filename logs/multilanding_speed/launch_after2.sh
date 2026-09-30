#!/bin/bash
# 変更後 (snap_after2 に固定した現行ツリー) の計装付き再生を順次実行する
R=/mnt/c/Users/ryouj/.codex/worktrees/exev/snap_after2
S=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed
bash $S/run_variant.sh $R after2 sources q_7gc4TgFig review fcXG83vInDY mia8KCjr52g zenchi
bash $S/run_variant.sh $R after2 live b18_stall b18_run5 b20_on
