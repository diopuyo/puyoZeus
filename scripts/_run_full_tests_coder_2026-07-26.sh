#!/bin/bash
cd <home>/.gemini/antigravity/scratch/puyo_analyzer
mkdir -p logs
PYTHONPATH=. ./venv/bin/python -m pytest tests/ -q > logs/_coder_full_test_run_2026-07-26.log 2>&1
