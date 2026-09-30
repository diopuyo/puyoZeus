#!/bin/bash
# 最終コードでの出力同一確認: gate exact (保存記録5本) → live 3 本 (変更後、現行ツリー)
S=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed
R=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
bash $S/run_gate.sh exact
VARIANT=exact bash $S/run_variant.sh $R after3 live b18_stall b18_run5 b20_on
