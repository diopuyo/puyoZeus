#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
setsid -f bash -c "scripts/_run_full_game9_c62_2026-07-23.sh > logs/full_game9_c62_2026-07-23.log 2>&1 < /dev/null"
