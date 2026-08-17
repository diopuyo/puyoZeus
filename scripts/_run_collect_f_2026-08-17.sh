#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
PYTHONPATH=. ./venv/bin/python -u -m scripts._collect_yardstick_v2_f_2026-08-17 > logs/collect_yardstick_v2_f_2026-08-17.log 2>&1
echo "EXIT_CODE=$?" >> logs/collect_yardstick_v2_f_2026-08-17.log
