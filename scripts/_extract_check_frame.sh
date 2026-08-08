#!/bin/bash
set -eu
cd /mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer
FF="$(PYTHONPATH=. ./venv/bin/python -c 'from src.video_compositer import VideoCompositor; print(VideoCompositor._resolve_ffmpeg_bin())')"
mkdir -p /tmp/check_frames
"${FF}" -y -ss "${2}" -i "${1}" -frames:v 1 "/tmp/check_frames/${3}.png" -loglevel error
mkdir -p /mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/ec05099f-6bfa-4f37-83fa-18a28fb06529/scratchpad/check_frames
cp "/tmp/check_frames/${3}.png" /mnt/c/Users/ryouj/AppData/Local/Temp/claude/C--Users-ryouj--gemini-antigravity-scratch-puyo-analyzer/ec05099f-6bfa-4f37-83fa-18a28fb06529/scratchpad/check_frames/
