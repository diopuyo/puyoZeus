#!/bin/bash
# 「変更後」の計装付き再生 (現行ツリー・HEAD 証明器、nice19・1 プロセス)。使い方: run_after.sh <OUT名> live|sources [名前...] ; 追加引数は EXTRA 環境変数
NAME=$1; MODE=$2; shift 2
bash /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed/run_variant.sh /mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer $NAME $MODE "$@"
