#!/bin/bash
# 最終コードで live 3 本 (出力同一の確認 + 速度)
S=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed
R=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
VARIANT=exact bash $S/run_variant.sh $R after3 live b18_stall b18_run5 b20_on
