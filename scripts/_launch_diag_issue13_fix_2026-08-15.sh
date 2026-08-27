#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
setsid -f bash -c "PYTHONPATH=. ./venv/bin/python -m scripts._diag_issue13_fix_verify_2026-08-15 > logs/_diag_issue13_fix_verify_2026-08-15.log 2>&1 < /dev/null"
