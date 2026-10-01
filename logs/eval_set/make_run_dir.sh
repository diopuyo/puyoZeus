#!/bin/bash
# 実行用 cwd (exev の data/logs/models を読取専用で参照する)。出力は全て絶対パスで wt_evalset/logs/eval_set へ。
set -eu
E=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer
R=/mnt/d/puyo_analyzer/evalset_run
mkdir -p $R
cd $R
rm -f data logs models
ln -s $E/data data
ln -s $E/logs logs
ln -s $E/models models
ls -la $R
