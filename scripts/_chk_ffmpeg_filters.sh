#!/bin/bash
FF=/mnt/c/Users/ryouj/.gemini/antigravity/scratch/puyo_analyzer/venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2
"$FF" -filters > /tmp/_ff_filters.txt 2>&1
wc -l /tmp/_ff_filters.txt
grep -i text /tmp/_ff_filters.txt
