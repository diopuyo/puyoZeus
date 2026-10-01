#!/usr/bin/env bash
# 採点器が相対パスで読む既存の logs 入力 (読取専用) を、wt 側に無いものだけ exev からシンボリックリンクする。
set -eu
SRC=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs
DST=/mnt/d/puyo_analyzer/wt_switch/logs
for p in "$SRC"/*; do
  n=$(basename "$p")
  [ -e "$DST/$n" ] || [ -L "$DST/$n" ] || ln -s "$p" "$DST/$n"
done
ls -la "$DST" | grep -c -- '->'
