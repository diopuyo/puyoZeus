#!/bin/bash
# 基準 (変更前 = snap_base) の単独計測。比較を公平にするため変更後と同じ条件 (1 プロセス・nice19) で b20 を再生する。
S=/mnt/c/Users/ryouj/.codex/worktrees/exev/puyo_analyzer/logs/multilanding_speed
VARIANT=exact bash $S/run_variant.sh /mnt/c/Users/ryouj/.codex/worktrees/exev/snap_base clean_base live b20_on
