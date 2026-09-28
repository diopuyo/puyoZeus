#!/usr/bin/env bash
set -eu
cd /mnt/c/Users/ryouj/.codex/worktrees/rt/puyo_analyzer
bash scripts/run_live_b14_probe.sh logs/live_b14/b13_baseline --baseline --stages --end 3000.566 --measure-start 2880.55
bash scripts/run_live_b14_probe.sh logs/live_b14/b14_candidate --stages --end 3000.566 --measure-start 2880.55
bash scripts/run_live_b14_probe.sh logs/live_b14/b8_control --baseline --stages --start 2580 --end 2700
