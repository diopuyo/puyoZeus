#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
setsid -f bash -c "./venv/bin/python -u -m scripts._diag_w20w21_boundary_multisignal_c109_2026-08-17 > logs/_diag_w20w21_c109_2026-08-17.log 2>&1 < /dev/null"
