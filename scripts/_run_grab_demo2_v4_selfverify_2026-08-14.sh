#!/bin/bash
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer || exit 1
export PYTHONPATH=.
./venv/bin/python -u scripts/_grab_demo2_v4_selfverify_frames_2026-08-14.py
