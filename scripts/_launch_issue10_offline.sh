#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
setsid -f bash -c "nice -n 19 ./venv/bin/python -u -m scripts._diag_issue10_offline_2026-08-14 > logs/_diag_issue10_offline_2026-08-14b.log 2>&1 < /dev/null"
